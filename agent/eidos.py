"""EIDOS — Emergent Intelligence via Distributed Organisational Systems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agent.config import (
    ASSOCIATION_DECAY,
    ATTENTION_ALPHA,
    ATTENTION_BETA,
    BELIEF_CONSOLIDATION_LR,
    BELIEF_CONSOLIDATION_STEPS,
    CONSOLIDATION_PREVIEW_STEPS,
    HEBBIAN_LEARNING_RATE,
    HIDDEN_DIM,
    HYPOTHESIS_ASSOCIATION_BOOST,
    HYPOTHESIS_BLEND_WEIGHT,
    HYPOTHESIS_PERSISTENCE_STEPS,
    INPUT_DIM,
    EPISODIC_BUFFER_CAPACITY,
    PREDICTION_LEARNING_RATE,
    REASONING_ABSOLUTE_FLOOR,
    RECENT_HISTORY_WINDOW,
    SALIENCE_THRESHOLD,
    SLEEP_ASSOCIATION_BOOST,
    SLEEP_CONSOLIDATE_STEPS,
    SLEEP_CONSOLIDATE_TOP_K,
    SLEEP_REPLAY_COUNT,
    SLEEP_REPLAY_INTEGRATE_WEIGHT,
    SLOW_STORE_CONFLICT_RATIO,
    SURPRISE_MIN_HISTORY,
    SURPRISE_SPIKE_RATIO,
    SURPRISE_STD_MULTIPLIER,
    SURPRISE_WINDOW,
    WORKSPACE_CAPACITY,
)
from architecture.components.association_store import AssociationStore
from architecture.components.attention_gate import AttentionGate
from architecture.components.belief_graph import BeliefGraph
from architecture.components.episodic_buffer import EpisodicBuffer
from architecture.components.prediction_engine import PredictionEngine
from architecture.components.reasoning_loop import ReasoningLoop
from architecture.components.recovery_context import RecoveryContextTracker
from architecture.components.reward_signal import RewardSignal
from architecture.components.sleep_replay import SleepReplay
from architecture.components.surprise_detector import SurpriseDetector
from architecture.components.working_memory import WorkspaceBuffer


class EidosAgent:
    """Predictive Active Workspace agent composing all cognitive primitives."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        workspace_capacity: int = WORKSPACE_CAPACITY,
        seed: int | None = None,
        enable_reasoning: bool = True,
        apply_hypotheses: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.enable_reasoning = enable_reasoning
        self.apply_hypotheses = apply_hypotheses

        self.workspace = WorkspaceBuffer(capacity=workspace_capacity)
        self.prediction = PredictionEngine(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learning_rate=PREDICTION_LEARNING_RATE,
            seed=seed,
        )
        self.associations = AssociationStore(
            learning_rate=HEBBIAN_LEARNING_RATE,
            decay_factor=ASSOCIATION_DECAY,
        )
        self.attention = AttentionGate(alpha=ATTENTION_ALPHA, beta=ATTENTION_BETA)
        self.reward = RewardSignal()
        self.surprise = SurpriseDetector(
            window=SURPRISE_WINDOW,
            spike_ratio=SURPRISE_SPIKE_RATIO,
            std_multiplier=SURPRISE_STD_MULTIPLIER,
            min_history=SURPRISE_MIN_HISTORY,
        )
        self.reasoner = ReasoningLoop(
            self.workspace,
            self.associations,
            self.prediction,
            threshold=REASONING_ABSOLUTE_FLOOR,
            seed=seed,
        )

        self._concept_vectors: dict[str, np.ndarray] = {}
        self._belief_bias: np.ndarray | None = None
        self._belief_persistence: int = 0
        self._step_count = 0
        self._last_applied_hypothesis: dict[str, Any] | None = None
        self._recovery_context = RecoveryContextTracker(window=RECENT_HISTORY_WINDOW)
        self._current_goal: np.ndarray | None = None
        self.belief_graph = BeliefGraph()
        self.episodic_buffer = EpisodicBuffer(capacity=EPISODIC_BUFFER_CAPACITY)
        self._sleep = SleepReplay(seed=seed)
        self._last_sleep_result: dict[str, Any] | None = None

    def _raw_workspace_mean_vector(self) -> np.ndarray:
        """Workspace mean without applying or consuming belief bias."""
        slots = self.workspace.broadcast()
        vectors = []
        for slot in slots:
            content = slot["content"]
            if isinstance(content, np.ndarray):
                vectors.append(content)
            elif isinstance(content, dict):
                if "vector" in content:
                    vectors.append(np.asarray(content["vector"], dtype=np.float64))
                elif content.get("label") in self._concept_vectors:
                    vectors.append(self._concept_vectors[content["label"]])
        if not vectors:
            return np.zeros(self.input_dim)
        mean = np.mean(vectors, axis=0)
        if mean.shape[0] < self.input_dim:
            padded = np.zeros(self.input_dim)
            padded[: mean.shape[0]] = mean
            return padded
        return mean[: self.input_dim]

    def _prediction_context(self) -> np.ndarray:
        """Context vector for prediction — may include sustained belief bias."""
        base = self._raw_workspace_mean_vector()
        if self._belief_bias is not None and self._belief_persistence > 0:
            bias = self._belief_bias[: self.input_dim]
            blended = (
                (1.0 - HYPOTHESIS_BLEND_WEIGHT) * base
                + HYPOTHESIS_BLEND_WEIGHT * bias
            )
            self._belief_persistence -= 1
            if self._belief_persistence <= 0:
                self._belief_bias = None
            return blended
        return base

    def register_concept(self, label: str, vector: np.ndarray) -> None:
        self._concept_vectors[label] = np.asarray(vector, dtype=np.float64)

    def sleep(
        self,
        n_replays: int = SLEEP_REPLAY_COUNT,
        integrate_weight: float = SLEEP_REPLAY_INTEGRATE_WEIGHT,
    ) -> dict[str, Any]:
        """Offline consolidation: replay episodic traces into slow BeliefGraph."""
        result = self._sleep.run(
            episodic_buffer=self.episodic_buffer,
            belief_graph=self.belief_graph,
            associations=self.associations,
            prediction_engine=self.prediction,
            concept_vectors=self._concept_vectors,
            n_replays=n_replays,
            integrate_weight=integrate_weight,
            association_boost=SLEEP_ASSOCIATION_BOOST,
            consolidate_top_k=SLEEP_CONSOLIDATE_TOP_K,
            consolidate_steps=SLEEP_CONSOLIDATE_STEPS,
            consolidate_lr=BELIEF_CONSOLIDATION_LR,
        )
        self._last_sleep_result = result
        return result

    def _recent_label_count(self, label: str) -> int:
        return self._recovery_context.label_count(label)

    def _resolve_recovery_probe(
        self,
        recovery_probe: np.ndarray | None,
        input_label: str,
        workspace_labels: list[str],
    ) -> tuple[np.ndarray | None, str | None, str]:
        """Explicit probe overrides; merge recent episodic + slow BeliefGraph."""
        if recovery_probe is not None:
            return (
                np.asarray(recovery_probe, dtype=np.float64).flatten(),
                None,
                "explicit",
            )

        recent_probe, recent_label, recent_source = (
            self._recovery_context.infer_recovery_probe(
                self._concept_vectors,
                input_label,
                self.input_dim,
                workspace_labels=workspace_labels,
            )
        )

        belief_probe, belief_label = self.belief_graph.infer_recovery_probe(
            self._concept_vectors,
            self.input_dim,
        )

        if belief_probe is not None and recent_probe is not None:
            if belief_label != recent_label:
                belief_strength = self.belief_graph.strength(belief_label or "")
                recent_strength = self.belief_graph.strength(recent_label or "")
                recent_count = self._recent_label_count(recent_label or "")
                slow_wins = (
                    belief_strength > max(recent_strength, float(recent_count))
                    * SLOW_STORE_CONFLICT_RATIO
                )
                if slow_wins:
                    return belief_probe, belief_label, "belief_graph"
            return recent_probe, recent_label, recent_source

        if belief_probe is not None:
            return belief_probe, belief_label, "belief_graph"

        if recent_probe is not None:
            return recent_probe, recent_label, recent_source

        if self._current_goal is not None:
            goal = self._current_goal
            best_label = None
            best_sim = -1.0
            for name, vec in self._concept_vectors.items():
                min_len = min(len(goal), len(vec))
                g, v = goal[:min_len], vec[:min_len]
                norm_g, norm_v = np.linalg.norm(g), np.linalg.norm(v)
                if norm_g < 1e-10 or norm_v < 1e-10:
                    continue
                sim = float(np.dot(g, v) / (norm_g * norm_v))
                if sim > best_sim:
                    best_sim = sim
                    best_label = name
            if best_label is not None and best_sim > 0.3:
                return self._concept_vectors[best_label].copy(), best_label, "goal"

        return None, None, "none"

    def _apply_hypothesis(
        self, hypothesis: dict[str, Any], surprise_label: str | None = None
    ) -> bool:
        """Close the loop: write hypothesis back into workspace and associations."""
        associate = hypothesis.get("associate")
        concept = hypothesis.get("concept")
        applied = False

        if surprise_label and surprise_label not in self._concept_vectors:
            self.workspace.evict_by_label(surprise_label)

        if associate and associate in self._concept_vectors:
            vec = self._concept_vectors[associate].copy()
            self.workspace.insert(
                content={
                    "label": associate,
                    "vector": vec,
                    "belief": True,
                    "from_hypothesis": hypothesis["label"],
                },
                salience=min(0.95, hypothesis.get("confidence", 0.5) + 0.4),
                source="reasoning",
            )
            self._belief_bias = vec
            self._belief_persistence = HYPOTHESIS_PERSISTENCE_STEPS

            ctx = self._raw_workspace_mean_vector()[: self.hidden_dim]
            if np.linalg.norm(ctx) < 1e-3:
                ctx = vec[: self.hidden_dim]

            n_steps = BELIEF_CONSOLIDATION_STEPS
            consolidate_lr = BELIEF_CONSOLIDATION_LR
            if hypothesis.get("recovery_inference_source") == "belief_graph":
                n_steps = max(BELIEF_CONSOLIDATION_STEPS, SLEEP_CONSOLIDATE_STEPS * 3)
                consolidate_lr = BELIEF_CONSOLIDATION_LR * 1.5

            consolidated_error = self.prediction.consolidate_belief(
                target=vec,
                context=ctx,
                n_steps=n_steps,
                learning_rate=consolidate_lr,
            )
            hypothesis["consolidated_error"] = consolidated_error
            applied = True

        if concept and associate:
            self.associations.hebbian_update([concept, associate])
            graph = self.associations.to_dict()
            if concept in graph and associate in graph[concept]:
                graph[concept][associate] += HYPOTHESIS_ASSOCIATION_BOOST
                graph[associate][concept] = graph[concept][associate]
                self.associations.load_dict(graph)

        if not applied and hypothesis.get("perturbation") is not None:
            perturbation = np.asarray(hypothesis["perturbation"], dtype=np.float64)
            self._belief_bias = self._raw_workspace_mean_vector() + perturbation
            self._belief_persistence = HYPOTHESIS_PERSISTENCE_STEPS
            applied = True

        if applied:
            self._last_applied_hypothesis = hypothesis
        return applied

    def step(
        self,
        raw_input: np.ndarray,
        input_label: str,
        goal: np.ndarray | None = None,
        goal_achieved: bool = False,
        recovery_probe: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Single cognitive cycle."""
        self._step_count += 1
        raw_input = np.asarray(raw_input, dtype=np.float64).flatten()

        if goal is not None:
            self.attention.set_goal(goal)
            self._current_goal = np.asarray(goal, dtype=np.float64).flatten()

        workspace_mean = self._raw_workspace_mean_vector()
        prediction_context = self._prediction_context()
        signal = {
            "id": f"input_{self._step_count}",
            "content_vector": raw_input,
            "source": input_label,
        }
        scored = self.attention.score_signals([signal], workspace_mean=workspace_mean)
        top = scored[0]
        salience = top["salience"]

        if salience > SALIENCE_THRESHOLD:
            self.workspace.insert(
                content={"label": input_label, "vector": raw_input},
                salience=salience,
                source=input_label,
            )

        context = prediction_context[: self.hidden_dim]
        pred_result = self.prediction.predict(raw_input, context=context)
        prediction_error = pred_result["prediction_error"]

        self.surprise.update(prediction_error)

        active_labels = []
        for slot in self.workspace.broadcast():
            content = slot["content"]
            if isinstance(content, dict) and "label" in content:
                active_labels.append(content["label"])
            elif isinstance(content, str):
                active_labels.append(content)

        if active_labels:
            self.associations.hebbian_update(active_labels)
        self.associations.apply_decay()

        reward = self.reward.update(prediction_error, goal_achieved=goal_achieved)

        hypothesis = None
        hypothesis_applied = False
        reasoning_triggered = False
        inferred_label: str | None = None
        inference_source = "none"

        effective_probe, inferred_label, inference_source = self._resolve_recovery_probe(
            recovery_probe, input_label, active_labels
        )

        belief_dominant, _ = self.belief_graph.dominant_concept(
            registered=set(self._concept_vectors.keys())
        )
        cold_start_reason = (
            input_label not in self._concept_vectors
            and belief_dominant is not None
            and prediction_error >= REASONING_ABSOLUTE_FLOOR
        )

        if self.enable_reasoning and (
            self.reasoner.should_reason(prediction_error, self.surprise)
            or cold_start_reason
        ):
            reasoning_triggered = True
            hypothesis = self.reasoner.run(
                prediction_error,
                concept_vectors=self._concept_vectors,
                probe_input=raw_input,
                recovery_probe=effective_probe,
                context=context,
                preview_steps=CONSOLIDATION_PREVIEW_STEPS,
                preview_lr=BELIEF_CONSOLIDATION_LR,
            )
            if hypothesis and self.apply_hypotheses:
                hypothesis["recovery_inference_source"] = inference_source
                hypothesis_applied = self._apply_hypothesis(
                    hypothesis, surprise_label=input_label
                )

        self._recovery_context.record(input_label, raw_input)
        self.episodic_buffer.record(input_label, raw_input)

        return {
            "step": self._step_count,
            "salience": salience,
            "prediction_error": prediction_error,
            "reward": reward,
            "hypothesis": hypothesis,
            "hypothesis_applied": hypothesis_applied,
            "reasoning_triggered": reasoning_triggered,
            "belief_consolidated": (
                hypothesis.get("consolidated_error") is not None
                if hypothesis else False
            ),
            "inferred_recovery_label": inferred_label,
            "recovery_inference_source": inference_source,
            "surprise_ratio": self.surprise.surprise_ratio(prediction_error),
            "error_baseline": self.surprise.baseline(),
            "workspace_size": len(self.workspace),
        }

    def run_episode(
        self,
        inputs: list[dict[str, Any]],
        goal: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        log = []
        for inp in inputs:
            vector = np.asarray(inp["vector"], dtype=np.float64)
            label = inp.get("label", "unknown")
            achieved = inp.get("goal_achieved", False)
            result = self.step(vector, label, goal=goal, goal_achieved=achieved)
            log.append(result)
        return log

    def _serialise_content(self, content: Any) -> Any:
        if isinstance(content, np.ndarray):
            return content.tolist()
        if isinstance(content, dict):
            return {k: self._serialise_content(v) for k, v in content.items()}
        return content

    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        serialised_slots = []
        for slot in self.workspace.broadcast():
            serialised_slots.append({
                **slot,
                "content": self._serialise_content(slot["content"]),
            })
        state = {
            "version": "2.0",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "workspace_capacity": self.workspace.capacity,
            "enable_reasoning": self.enable_reasoning,
            "apply_hypotheses": self.apply_hypotheses,
            "workspace_slots": serialised_slots,
            "prediction_weights": self.prediction.get_weights_dict(),
            "associations": self.associations.to_dict(),
            "reward": self.reward.to_dict(),
            "surprise": self.surprise.to_dict(),
            "recovery_context": self._recovery_context.to_dict(),
            "belief_graph": self.belief_graph.to_dict(),
            "episodic_buffer": self.episodic_buffer.to_dict(),
            "concept_vectors": {k: v.tolist() for k, v in self._concept_vectors.items()},
            "step_count": self._step_count,
            "reasoning_trace": self.reasoner.get_trace(),
        }
        path.write_text(json.dumps(state, indent=2))

    def load_state(self, path: str | Path) -> None:
        path = Path(path)
        state = json.loads(path.read_text())

        self.input_dim = state["input_dim"]
        self.hidden_dim = state["hidden_dim"]
        self.enable_reasoning = state.get("enable_reasoning", True)
        self.apply_hypotheses = state.get("apply_hypotheses", True)
        self.workspace = WorkspaceBuffer(capacity=state["workspace_capacity"])
        for slot in state.get("workspace_slots", []):
            self.workspace.insert(
                content=slot["content"],
                salience=slot["salience"],
                source=slot["source"],
            )

        self.prediction.load_weights_dict(state["prediction_weights"])
        self.associations.load_dict(state["associations"])
        self.reward.load_dict(state["reward"])
        if "surprise" in state:
            self.surprise.load_dict(state["surprise"])
        if "recovery_context" in state:
            self._recovery_context.load_dict(state["recovery_context"])
        else:
            self._recovery_context.clear()
        if "belief_graph" in state:
            self.belief_graph.load_dict(state["belief_graph"])
        else:
            self.belief_graph.clear()
        if "episodic_buffer" in state:
            self.episodic_buffer.load_dict(state["episodic_buffer"])
        else:
            self.episodic_buffer.clear()
        self._concept_vectors = {
            k: np.array(v, dtype=np.float64)
            for k, v in state.get("concept_vectors", {}).items()
        }
        self._step_count = state.get("step_count", 0)

        self.reasoner = ReasoningLoop(
            self.workspace,
            self.associations,
            self.prediction,
            threshold=REASONING_ABSOLUTE_FLOOR,
        )
        for entry in state.get("reasoning_trace", []):
            self.reasoner._reasoning_trace.append(entry)

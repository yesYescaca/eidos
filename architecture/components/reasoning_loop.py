"""System 2 deliberate reasoning — hypothesis generation under surprise."""

from __future__ import annotations

from typing import Any

import numpy as np

from architecture.components.association_store import AssociationStore
from architecture.components.prediction_engine import PredictionEngine
from architecture.components.surprise_detector import SurpriseDetector
from architecture.components.working_memory import WorkspaceBuffer


class ReasoningLoop:
    """Triggered on relative surprise. Generates and scores hypotheses via consolidation preview."""

    def __init__(
        self,
        workspace: WorkspaceBuffer,
        associations: AssociationStore,
        prediction_engine: PredictionEngine,
        threshold: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.workspace = workspace
        self.associations = associations
        self.prediction_engine = prediction_engine
        self.threshold = threshold
        self._rng = np.random.default_rng(seed)
        self._reasoning_trace: list[dict[str, Any]] = []

    def should_reason(
        self,
        current_error: float,
        surprise_detector: SurpriseDetector,
    ) -> bool:
        if current_error < self.threshold:
            return False
        return surprise_detector.is_surprising(current_error)

    def _extract_concepts(self) -> list[str]:
        concepts = []
        for slot in self.workspace.broadcast():
            content = slot["content"]
            if isinstance(content, str):
                concepts.append(content)
            elif isinstance(content, dict) and "label" in content:
                concepts.append(str(content["label"]))
            else:
                concepts.append(str(slot["id"]))
        return list(dict.fromkeys(concepts))

    def _workspace_input_vector(self) -> np.ndarray:
        slots = self.workspace.broadcast()
        if not slots:
            return np.zeros(self.prediction_engine.input_dim)

        vectors = []
        for slot in slots:
            content = slot["content"]
            if isinstance(content, np.ndarray):
                vectors.append(content)
            elif isinstance(content, dict) and "vector" in content:
                vectors.append(np.asarray(content["vector"], dtype=np.float64))

        if not vectors:
            return self._rng.normal(0, 0.1, self.prediction_engine.input_dim)

        pooled = np.mean(vectors, axis=0)
        dim = self.prediction_engine.input_dim
        if pooled.shape[0] < dim:
            padded = np.zeros(dim)
            padded[: pooled.shape[0]] = pooled
            return padded
        return pooled[:dim]

    def _score_consolidation_preview(
        self,
        target: np.ndarray,
        probe_input: np.ndarray,
        context: np.ndarray | None,
        preview_steps: int,
        preview_lr: float,
    ) -> float:
        return self.prediction_engine.preview_consolidation(
            target=target,
            probe_input=probe_input,
            context=context,
            n_steps=preview_steps,
            learning_rate=preview_lr,
        )

    def run(
        self,
        current_error: float,
        concept_vectors: dict[str, np.ndarray] | None = None,
        probe_input: np.ndarray | None = None,
        recovery_probe: np.ndarray | None = None,
        context: np.ndarray | None = None,
        preview_steps: int = 8,
        preview_lr: float = 0.02,
    ) -> dict[str, Any] | None:
        if current_error < self.threshold:
            return None

        active_concepts = self._extract_concepts()
        base_input = self._workspace_input_vector()
        concept_vectors = concept_vectors or {}
        scoring_probe = recovery_probe
        if scoring_probe is None:
            scoring_probe = probe_input
        if scoring_probe is None:
            scoring_probe = base_input
        scoring_probe = np.asarray(scoring_probe, dtype=np.float64)

        if context is not None:
            ctx = np.asarray(context, dtype=np.float64).flatten()
            if ctx.shape[0] > self.prediction_engine.hidden_dim:
                ctx = ctx[: self.prediction_engine.hidden_dim]
            elif ctx.shape[0] < self.prediction_engine.hidden_dim:
                padded = np.zeros(self.prediction_engine.hidden_dim)
                padded[: ctx.shape[0]] = ctx
                ctx = padded
        else:
            ctx = scoring_probe[: self.prediction_engine.hidden_dim]

        hypotheses: list[dict[str, Any]] = []
        seen_labels: set[str] = set()

        # v1.3: score registered concepts — probe = recovery target when provided
        for concept, vec in concept_vectors.items():
            label = f"explain:{concept}"
            if label in seen_labels:
                continue
            seen_labels.add(label)
            preview_error = self._score_consolidation_preview(
                vec, scoring_probe, ctx, preview_steps, preview_lr
            )
            hypotheses.append({
                "label": label,
                "concept": concept,
                "associate": concept,
                "predicted_error": preview_error,
                "preview_recovery_error": preview_error,
                "hypothesis_type": "consolidation_preview",
                "scoring": "consolidation_preview",
            })

        # Association hypotheses — preview consolidation toward associate
        for concept in active_concepts:
            for associate, weight in self.associations.get_associates(concept, top_k=5):
                if associate not in concept_vectors:
                    continue
                label = f"{concept}->{associate}"
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                vec = concept_vectors[associate]
                preview_error = self._score_consolidation_preview(
                    vec, scoring_probe, ctx, preview_steps, preview_lr
                )
                hypotheses.append({
                    "label": label,
                    "concept": concept,
                    "associate": associate,
                    "association_weight": weight,
                    "predicted_error": preview_error,
                    "preview_recovery_error": preview_error,
                    "hypothesis_type": "association_preview",
                    "scoring": "consolidation_preview",
                })

        # Path hypotheses — preview toward path endpoint
        unique_active = list(dict.fromkeys(active_concepts))
        for i, source in enumerate(unique_active):
            for target in unique_active[i + 1 :]:
                path = self.associations.get_strongest_path(source, target)
                if not path or len(path) < 2:
                    continue
                if not all(p in concept_vectors for p in path):
                    continue
                label = "->".join(path)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                endpoint = path[-1]
                vec = concept_vectors[endpoint]
                preview_error = self._score_consolidation_preview(
                    vec, scoring_probe, ctx, preview_steps, preview_lr
                )
                hypotheses.append({
                    "label": label,
                    "concept": source,
                    "associate": endpoint,
                    "path": path,
                    "predicted_error": preview_error,
                    "preview_recovery_error": preview_error,
                    "hypothesis_type": "path_preview",
                    "scoring": "consolidation_preview",
                })

        if not hypotheses:
            return None

        best = min(hypotheses, key=lambda h: h["predicted_error"])
        confidence = max(
            0.0,
            min(1.0, 1.0 - best["predicted_error"] / max(current_error, 1e-6)),
        )

        result = {
            "label": best["label"],
            "confidence": confidence,
            "predicted_error": best["predicted_error"],
            "preview_recovery_error": best["preview_recovery_error"],
            "current_error": current_error,
            "hypotheses_evaluated": len(hypotheses),
            "concept": best.get("concept"),
            "associate": best.get("associate"),
            "hypothesis_type": best.get("hypothesis_type"),
            "scoring": best.get("scoring"),
            "path": best.get("path"),
        }

        trace_entry = {
            "trigger_error": current_error,
            "active_concepts": active_concepts,
            "hypotheses": [
                {
                    "label": h["label"],
                    "predicted_error": h["predicted_error"],
                    "type": h.get("hypothesis_type"),
                    "scoring": h.get("scoring"),
                }
                for h in sorted(hypotheses, key=lambda x: x["predicted_error"])
            ],
            "selected": result,
        }
        self._reasoning_trace.append(trace_entry)
        return result

    def get_trace(self) -> list[dict[str, Any]]:
        return list(self._reasoning_trace)

    def clear_trace(self) -> None:
        self._reasoning_trace.clear()

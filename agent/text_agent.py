"""EidosTextAgent — language grounding wrapper over EidosAgent (v5.0+)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agent.eidos import EidosAgent
from architecture.bridge.embedding_factory import create_grounding
from architecture.bridge.text_grounding import TextGroundingBridge
from architecture.gate.gate_policy import GatePolicy


def interpret_text_decision(step_result: dict[str, Any]) -> str:
    """Map cognitive step to text-facing decision (via GatePolicy)."""
    return GatePolicy.decision_from_step(step_result)


class EidosTextAgent:
    """
    Wraps EidosAgent with a text grounding bridge.

    v6.0: optional `embedding_backend` ('hash' | 'sbert') and GatePolicy on steps.
    """

    def __init__(
        self,
        grounding: TextGroundingBridge | None = None,
        embedding_backend: str = "hash",
        gate_policy: GatePolicy | None = None,
        **agent_kwargs: Any,
    ) -> None:
        self.grounding = grounding or create_grounding(embedding_backend)  # type: ignore[arg-type]
        self.agent = EidosAgent(**agent_kwargs)
        self._text_concepts: dict[str, str] = {}
        self.gate_policy = gate_policy or GatePolicy()
        self.embedding_backend = embedding_backend

    def register_text_concept(self, label: str, phrase: str) -> np.ndarray:
        vector = self.grounding.embed(phrase)
        self.agent.register_concept(label, vector)
        self._text_concepts[label] = phrase
        return vector

    def embed(self, text: str) -> np.ndarray:
        return self.grounding.embed(text)

    def step_text(
        self,
        text: str,
        input_label: str | None = None,
        goal_text: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        vector = self.grounding.embed(text)
        label = input_label or "text_input"
        goal = self.grounding.embed(goal_text) if goal_text else kwargs.pop("goal", None)

        result = self.agent.step(vector, label, goal=goal, **kwargs)
        result["source_text"] = text
        gate_eval = self.gate_policy.evaluate_step(result)
        result["text_decision"] = gate_eval.decision
        result["gate_evaluation"] = {
            "scores": gate_eval.scores,
            "reasons": gate_eval.reasons,
        }
        if goal_text is not None:
            result["goal_text"] = goal_text
        return result

    def sleep(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.agent.sleep(*args, **kwargs)

    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        self.agent.save_state(path)
        state = json.loads(path.read_text())
        state["version"] = "7.6"
        state["text_concepts"] = dict(self._text_concepts)
        state["embedding_backend"] = self.embedding_backend
        path.write_text(json.dumps(state, indent=2))

    def load_state(self, path: str | Path) -> None:
        path = Path(path)
        state = json.loads(path.read_text())
        self.agent.load_state(path)
        self._text_concepts = dict(state.get("text_concepts", {}))
        self.embedding_backend = state.get("embedding_backend", "hash")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

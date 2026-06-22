"""EidosTextAgent — language grounding wrapper over EidosAgent (v5.0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agent.eidos import EidosAgent
from architecture.bridge.text_grounding import TextGroundingBridge


def interpret_text_decision(step_result: dict[str, Any]) -> str:
    """Map cognitive step flags to a text-facing decision."""
    flags = step_result.get("meta_cognition_flags", [])
    action = step_result.get("selected_action")

    if step_result.get("active_sleep_performed"):
        return "sleep"
    if action and str(action).startswith("probe:"):
        return "probe"
    if "hypothesis_deferred" in flags:
        return "defer"
    if "ambiguous_hypothesis" in flags or "low_confidence" in flags:
        return "clarify"
    if step_result.get("hypothesis_applied"):
        return "commit"
    return "observe"


class EidosTextAgent:
    """
    Wraps EidosAgent with a text grounding bridge.

    Phrases are embedded to 64-d vectors; all PAW mechanisms operate unchanged.
    """

    def __init__(
        self,
        grounding: TextGroundingBridge | None = None,
        **agent_kwargs: Any,
    ) -> None:
        self.grounding = grounding or TextGroundingBridge()
        self.agent = EidosAgent(**agent_kwargs)
        self._text_concepts: dict[str, str] = {}

    def register_text_concept(self, label: str, phrase: str) -> np.ndarray:
        """Register a concept from natural-language text."""
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
        """Run one cognitive cycle on embedded text."""
        vector = self.grounding.embed(text)
        label = input_label or "text_input"
        goal = self.grounding.embed(goal_text) if goal_text else kwargs.pop("goal", None)

        result = self.agent.step(vector, label, goal=goal, **kwargs)
        result["source_text"] = text
        result["text_decision"] = interpret_text_decision(result)
        if goal_text is not None:
            result["goal_text"] = goal_text
        return result

    def sleep(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.agent.sleep(*args, **kwargs)

    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        self.agent.save_state(path)
        state = json.loads(path.read_text())
        state["version"] = "5.0"
        state["text_concepts"] = dict(self._text_concepts)
        path.write_text(json.dumps(state, indent=2))

    def load_state(self, path: str | Path) -> None:
        path = Path(path)
        state = json.loads(path.read_text())
        self.agent.load_state(path)
        self._text_concepts = dict(state.get("text_concepts", {}))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

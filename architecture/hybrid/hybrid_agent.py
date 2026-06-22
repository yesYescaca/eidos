"""HybridEidosAgent — LLM generation gated by EIDOS cognition."""

from __future__ import annotations

from typing import Any, Literal

from agent.config import TEXT_ANOMALY_LABEL
from agent.text_agent import EidosTextAgent, interpret_text_decision
from architecture.hybrid.llm_backend import LanguageModelBackend, MockLanguageModel

GateDecision = Literal["commit", "defer", "clarify", "probe", "sleep", "observe"]

# Priority order — higher index wins when merging monitor steps
_DECISION_PRIORITY: dict[str, int] = {
    "observe": 0,
    "commit": 1,
    "probe": 2,
    "clarify": 3,
    "defer": 4,
    "sleep": 5,
}


def merge_decisions(*decisions: str) -> GateDecision:
    """Conservative merge: pick the most cautious decision."""
    best = "observe"
    best_rank = -1
    for d in decisions:
        rank = _DECISION_PRIORITY.get(d, 0)
        if rank > best_rank:
            best = d  # type: ignore[assignment]
            best_rank = rank
    return best  # type: ignore[return-value]


def gate_response(
    decision: GateDecision,
    draft: str,
    probe_concept: str | None = None,
    probe_phrase: str | None = None,
) -> str:
    """Map gate decision to user-facing text."""
    if decision == "commit":
        return draft
    if decision == "probe" and probe_phrase:
        return (
            f"I need to verify context first. Probing concept '{probe_concept}': "
            f"{probe_phrase}"
        )
    if decision == "defer":
        return (
            "I'm not confident enough to answer yet. "
            "Deferring until I can consolidate more context."
        )
    if decision == "clarify":
        return (
            "This could mean more than one thing. "
            "Can you clarify which scenario you mean?"
        )
    if decision == "sleep":
        return (
            "I need a moment to consolidate memory before answering. "
            "(Sleep replay triggered.)"
        )
    return draft


class HybridEidosAgent:
    """
    System 1 (LLM) + System 2 (EIDOS) hybrid.

    The LLM proposes; EIDOS monitors question and draft, then gates output.
    """

    def __init__(
        self,
        text_agent: EidosTextAgent | None = None,
        llm: LanguageModelBackend | None = None,
        enable_gate: bool = True,
        **text_agent_kwargs: Any,
    ) -> None:
        self.text = text_agent or EidosTextAgent(**text_agent_kwargs)
        self.llm = llm or MockLanguageModel()
        self.enable_gate = enable_gate

    def register_domain(self, concepts: dict[str, str]) -> None:
        """Register labelled phrases for the session domain."""
        for label, phrase in concepts.items():
            self.text.register_text_concept(label, phrase)

    def warm_session(self, training: list[tuple[str, str]], n_each: int = 1) -> None:
        """Optional exposure steps before Q&A."""
        for label, phrase in training:
            for _ in range(n_each):
                self.text.step_text(phrase, label)

    def respond(
        self,
        user_text: str,
        goal_text: str | None = None,
        prompt_template: str | None = None,
    ) -> dict[str, Any]:
        """
        Full hybrid cycle: monitor → generate → monitor → gate.

        Returns dict with draft, final_response, gate_decision, and step traces.
        """
        self.text.agent.workspace.clear()
        self.text.agent._recovery_context.clear()
        self.text.agent.surprise._history.clear()

        question_step = self.text.step_text(
            user_text,
            TEXT_ANOMALY_LABEL,
            goal_text=goal_text,
        )

        template = prompt_template or "Question: {q}\nAnswer:"
        prompt = template.format(q=user_text)
        draft = self.llm.generate(prompt)

        draft_step = self.text.step_text(
            draft,
            "llm_draft",
            goal_text=goal_text,
        )

        q_decision = interpret_text_decision(question_step)
        d_decision = interpret_text_decision(draft_step)
        gate_decision = merge_decisions(q_decision, d_decision)

        probe_concept = None
        probe_phrase = None
        action = draft_step.get("selected_action") or question_step.get("selected_action")
        if action and str(action).startswith("probe:"):
            probe_concept = str(action).split(":", 1)[1]
            probe_phrase = self.text._text_concepts.get(probe_concept)

        if not self.enable_gate:
            gate_decision = "commit"

        final = gate_response(gate_decision, draft, probe_concept, probe_phrase)

        return {
            "user_text": user_text,
            "goal_text": goal_text,
            "llm_draft": draft,
            "final_response": final,
            "gate_decision": gate_decision,
            "question_decision": q_decision,
            "draft_decision": d_decision,
            "question_step": question_step,
            "draft_step": draft_step,
            "gated": self.enable_gate and gate_decision != "commit",
        }

"""HybridEidosAgent — LLM generation gated by unified GatePolicy (v6.0)."""

from __future__ import annotations

from typing import Any

from agent.config import TEXT_ANOMALY_LABEL
from agent.text_agent import EidosTextAgent
from architecture.gate.gate_policy import GateEvaluation, GatePolicy, gate_response
from architecture.hybrid.llm_backend import LanguageModelBackend, MockLanguageModel


def merge_decisions(*decisions: str) -> str:
    """Backward-compatible alias for v5.1 tests."""
    return GatePolicy.merge_cognitive(*decisions)


class HybridEidosAgent:
    """
    System 1 (LLM) + System 2 (EIDOS) hybrid.

    v6.0: `GatePolicy` fuses cognitive steps + draft–goal text alignment.
    """

    def __init__(
        self,
        text_agent: EidosTextAgent | None = None,
        llm: LanguageModelBackend | None = None,
        enable_gate: bool = True,
        use_unified_gate: bool = True,
        gate_policy: GatePolicy | None = None,
        **text_agent_kwargs: Any,
    ) -> None:
        self.text = text_agent or EidosTextAgent(**text_agent_kwargs)
        self.llm = llm or MockLanguageModel()
        self.enable_gate = enable_gate
        self.use_unified_gate = use_unified_gate
        self.gate_policy = gate_policy or GatePolicy()

    def register_domain(self, concepts: dict[str, str]) -> None:
        for label, phrase in concepts.items():
            self.text.register_text_concept(label, phrase)

    def warm_session(self, training: list[tuple[str, str]], n_each: int = 1) -> None:
        for label, phrase in training:
            for _ in range(n_each):
                self.text.step_text(phrase, label)

    def respond(
        self,
        user_text: str,
        goal_text: str | None = None,
        prompt_template: str | None = None,
    ) -> dict[str, Any]:
        self.text.agent.workspace.clear()
        self.text.agent._recovery_context.clear()
        self.text.agent.surprise._history.clear()

        question_step = self.text.step_text(
            user_text,
            TEXT_ANOMALY_LABEL,
            goal_text=goal_text,
        )

        template = prompt_template or "Question: {q}\nAnswer:"
        draft = self.llm.generate(template.format(q=user_text))

        draft_step = self.text.step_text(
            draft,
            "llm_draft",
            goal_text=goal_text,
        )

        if self.use_unified_gate:
            evaluation = self.gate_policy.evaluate(
                question_step,
                draft_step,
                user_text=user_text,
                draft_text=draft,
                goal_text=goal_text,
                grounding=self.text.grounding,
                text_concepts=self.text._text_concepts,
            )
        else:
            legacy = GatePolicy.merge_cognitive(
                GatePolicy.decision_from_step(question_step),
                GatePolicy.decision_from_step(draft_step),
            )
            evaluation = GateEvaluation(
                decision=legacy,
                cognitive_decision=legacy,
                reasons=["legacy_v5_merge"],
            )

        if not self.enable_gate:
            evaluation = GateEvaluation(
                decision="commit",
                cognitive_decision=evaluation.cognitive_decision,
                scores=evaluation.scores,
                reasons=evaluation.reasons + ["gate_disabled"],
            )

        probe_concept = None
        probe_phrase = None
        action = draft_step.get("selected_action") or question_step.get("selected_action")
        if action and str(action).startswith("probe:"):
            probe_concept = str(action).split(":", 1)[1]
            probe_phrase = self.text._text_concepts.get(probe_concept)

        final = gate_response(evaluation, draft, probe_concept, probe_phrase)

        return {
            "user_text": user_text,
            "goal_text": goal_text,
            "llm_draft": draft,
            "final_response": final,
            "gate_decision": evaluation.decision,
            "gate_evaluation": {
                "cognitive_decision": evaluation.cognitive_decision,
                "scores": evaluation.scores,
                "reasons": evaluation.reasons,
            },
            "question_step": question_step,
            "draft_step": draft_step,
            "gated": self.enable_gate and evaluation.decision != "commit",
            "use_unified_gate": self.use_unified_gate,
        }

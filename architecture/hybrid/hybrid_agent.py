"""HybridEidosAgent — LLM generation gated by unified GatePolicy (v6.0+)."""

from __future__ import annotations

from typing import Any

from agent.config import META_INJECTION_MAX_ROUNDS, TEXT_ANOMALY_LABEL
from agent.text_agent import EidosTextAgent
from architecture.bridge.embedding_factory import (
    create_hybrid_grounding,
    resolve_hybrid_embedding_backend,
)
from architecture.gate.gate_policy import GateEvaluation, GatePolicy, gate_response
from architecture.hybrid.llm_backend import LanguageModelBackend, MockLanguageModel
from architecture.hybrid.metacognitive_prompt import (
    build_meta_injection,
    build_revision_prompt,
)


def merge_decisions(*decisions: str) -> str:
    """Backward-compatible alias for v5.1 tests."""
    return GatePolicy.merge_cognitive(*decisions)


class HybridEidosAgent:
    """
    System 1 (LLM) + System 2 (EIDOS) hybrid.

    v7.0: optional metacognitive injection loop + SBERT-default grounding.
    """

    def __init__(
        self,
        text_agent: EidosTextAgent | None = None,
        llm: LanguageModelBackend | None = None,
        enable_gate: bool = True,
        use_unified_gate: bool = True,
        enable_meta_injection: bool = False,
        max_revision_rounds: int = META_INJECTION_MAX_ROUNDS,
        gate_policy: GatePolicy | None = None,
        hybrid_embedding: bool = True,
        **text_agent_kwargs: Any,
    ) -> None:
        if text_agent is None:
            if hybrid_embedding and "embedding_backend" not in text_agent_kwargs:
                if "grounding" not in text_agent_kwargs:
                    text_agent_kwargs["grounding"] = create_hybrid_grounding()
                text_agent_kwargs["embedding_backend"] = resolve_hybrid_embedding_backend()
            self.text = EidosTextAgent(**text_agent_kwargs)
        else:
            self.text = text_agent
        self.llm = llm or MockLanguageModel()
        self.enable_gate = enable_gate
        self.use_unified_gate = use_unified_gate
        self.enable_meta_injection = enable_meta_injection
        self.max_revision_rounds = max_revision_rounds
        self.gate_policy = gate_policy or GatePolicy()

    def register_domain(self, concepts: dict[str, str]) -> None:
        for label, phrase in concepts.items():
            self.text.register_text_concept(label, phrase)

    def warm_session(self, training: list[tuple[str, str]], n_each: int = 1) -> None:
        for label, phrase in training:
            for _ in range(n_each):
                self.text.step_text(phrase, label)

    def _evaluate(
        self,
        question_step: dict[str, Any],
        draft_step: dict[str, Any],
        *,
        user_text: str,
        draft: str,
        goal_text: str | None,
    ) -> GateEvaluation:
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
        return evaluation

    def respond(
        self,
        user_text: str,
        goal_text: str | None = None,
        prompt_template: str | None = None,
        reset: bool = True,
    ) -> dict[str, Any]:
        if reset:
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

        evaluation = self._evaluate(
            question_step,
            draft_step,
            user_text=user_text,
            draft=draft,
            goal_text=goal_text,
        )

        revision_rounds: list[dict[str, Any]] = []
        rounds = 0
        while (
            self.enable_meta_injection
            and self.enable_gate
            and evaluation.decision != "commit"
            and rounds < self.max_revision_rounds
        ):
            injection = build_meta_injection(evaluation, question_step, draft_step)
            revision_prompt = build_revision_prompt(user_text, draft, injection)
            revised = self.llm.generate(revision_prompt)
            revision_rounds.append(
                {
                    "round": rounds + 1,
                    "prior_draft": draft,
                    "injection": injection,
                    "revised_draft": revised,
                }
            )
            draft = revised
            draft_step = self.text.step_text(
                draft,
                "llm_draft",
                goal_text=goal_text,
            )
            evaluation = self._evaluate(
                question_step,
                draft_step,
                user_text=user_text,
                draft=draft,
                goal_text=goal_text,
            )
            rounds += 1

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
            "meta_injection": self.enable_meta_injection,
            "revision_rounds": revision_rounds,
        }

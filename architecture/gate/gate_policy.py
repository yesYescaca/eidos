"""Unified gate policy — cognitive + text alignment (v6.0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent.config import (
    GATE_CONCEPT_AMBIGUITY_EPS,
    GATE_DRAFT_CONCEPT_MISMATCH,
    GATE_MIN_DRAFT_GOAL_ALIGN,
    GATE_QUESTION_GOAL_CLEAR,
)
from architecture.gate.underdetermination import has_underdetermination_markers

GateDecision = Literal["commit", "defer", "clarify", "probe", "sleep", "observe"]

_DECISION_PRIORITY: dict[str, int] = {
    "observe": 0,
    "commit": 1,
    "probe": 2,
    "clarify": 3,
    "defer": 4,
    "sleep": 5,
}


@dataclass
class GateEvaluation:
    decision: GateDecision
    cognitive_decision: GateDecision
    scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


class GatePolicy:
    """
    Fuses meta-cognition, active inference, and text-alignment into one gate.

    Inspired by dual-process monitoring (Kahneman) and metacognitive control
    (Nelson & Narens): fast outputs are vetoed when alignment checks fail.
    """

    def __init__(
        self,
        min_draft_goal_align: float = GATE_MIN_DRAFT_GOAL_ALIGN,
        concept_ambiguity_eps: float = GATE_CONCEPT_AMBIGUITY_EPS,
        draft_concept_mismatch: bool = GATE_DRAFT_CONCEPT_MISMATCH,
        question_goal_clear: float = GATE_QUESTION_GOAL_CLEAR,
        factual_mode: bool = False,
        require_underdetermination_to_abstain: bool = False,
    ) -> None:
        self.min_draft_goal_align = min_draft_goal_align
        self.concept_ambiguity_eps = concept_ambiguity_eps
        self.draft_concept_mismatch = draft_concept_mismatch
        self.question_goal_clear = question_goal_clear
        self.factual_mode = factual_mode
        self.require_underdetermination_to_abstain = require_underdetermination_to_abstain

    @staticmethod
    def decision_from_step(step_result: dict[str, Any]) -> GateDecision:
        """Cognitive decision from a single PAW step."""
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

    @staticmethod
    def merge_cognitive(*decisions: str) -> GateDecision:
        """Conservative merge across steps."""
        best: GateDecision = "observe"
        best_rank = -1
        for d in decisions:
            rank = _DECISION_PRIORITY.get(d, 0)
            if rank > best_rank:
                best = d  # type: ignore[assignment]
                best_rank = rank
        return best

    def _concept_ambiguity(
        self,
        user_text: str,
        text_concepts: dict[str, str],
        grounding: Any,
    ) -> tuple[float, float]:
        if not user_text or not text_concepts or grounding is None:
            return float("inf"), 0.0

        ranked = sorted(
            (
                (label, grounding.similarity(user_text, phrase))
                for label, phrase in text_concepts.items()
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        if len(ranked) < 2:
            return float("inf"), ranked[0][1] if ranked else 0.0
        gap = ranked[0][1] - ranked[1][1]
        return gap, ranked[0][1]

    def _draft_concept_mismatch(
        self,
        draft_text: str,
        goal_text: str,
        text_concepts: dict[str, str],
        grounding: Any,
    ) -> tuple[bool, str | None]:
        """True when draft best-matches a different concept than the goal."""
        if (
            not self.draft_concept_mismatch
            or not draft_text
            or not goal_text
            or not text_concepts
            or grounding is None
        ):
            return False, None

        draft_ranked = sorted(
            (
                (label, float(grounding.similarity(draft_text, phrase)))
                for label, phrase in text_concepts.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        goal_ranked = sorted(
            (
                (label, float(grounding.similarity(goal_text, phrase)))
                for label, phrase in text_concepts.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not draft_ranked or not goal_ranked:
            return False, None

        draft_label, draft_sim = draft_ranked[0]
        goal_label, _ = goal_ranked[0]
        if draft_label == goal_label:
            return False, None

        draft_to_goal = float(grounding.similarity(draft_text, goal_text))
        if draft_to_goal >= self.min_draft_goal_align:
            return False, None
        if draft_sim > draft_to_goal:
            return True, draft_label
        return False, None

    def _draft_aligns_truth_concept(
        self,
        draft_text: str,
        text_concepts: dict[str, str],
        grounding: Any,
    ) -> bool:
        """True when draft is closer to truth than falsehood (misconception items)."""
        if not draft_text or grounding is None:
            return False
        truth_phrase = text_concepts.get("truth")
        false_phrase = text_concepts.get("falsehood")
        if not truth_phrase or not false_phrase:
            return False
        truth_sim = float(grounding.similarity(draft_text, truth_phrase))
        false_sim = float(grounding.similarity(draft_text, false_phrase))
        return truth_sim >= self.min_draft_goal_align and truth_sim > false_sim

    def _should_abstain_on_misalignment(
        self,
        *,
        user_text: str,
        misaligned: bool,
        draft_text: str,
        text_concepts: dict[str, str],
        grounding: Any | None,
    ) -> bool:
        if not misaligned:
            return False
        if self.factual_mode and self.require_underdetermination_to_abstain:
            if has_underdetermination_markers(user_text):
                return True
            if self._draft_aligns_truth_concept(
                draft_text, text_concepts, grounding
            ):
                return False
            return False
        return True

    def evaluate(
        self,
        question_step: dict[str, Any],
        draft_step: dict[str, Any] | None = None,
        *,
        user_text: str | None = None,
        draft_text: str | None = None,
        goal_text: str | None = None,
        grounding: Any | None = None,
        text_concepts: dict[str, str] | None = None,
    ) -> GateEvaluation:
        """Unified gate across question + optional LLM draft."""
        q_dec = self.decision_from_step(question_step)
        d_dec = self.decision_from_step(draft_step) if draft_step else "observe"
        cognitive = self.merge_cognitive(q_dec, d_dec)

        scores: dict[str, float] = {
            "question_surprise_ratio": float(
                question_step.get("surprise_ratio", 0.0) or 0.0
            ),
            "draft_prediction_error": float(
                draft_step.get("prediction_error", 0.0) if draft_step else 0.0
            ),
        }
        reasons: list[str] = [f"cognitive:{cognitive}"]

        draft_goal_align = 1.0
        misaligned = False
        if grounding and draft_text and goal_text:
            draft_goal_align = grounding.similarity(draft_text, goal_text)
            scores["draft_goal_alignment"] = draft_goal_align
            misaligned = draft_goal_align < self.min_draft_goal_align

        concept_mismatch, mismatch_label = self._draft_concept_mismatch(
            draft_text or "",
            goal_text or "",
            text_concepts or {},
            grounding,
        )
        if concept_mismatch:
            misaligned = True
            scores["draft_concept_mismatch"] = 1.0

        concept_gap, top_concept_sim = self._concept_ambiguity(
            user_text or "", text_concepts or {}, grounding
        )
        scores["concept_gap"] = concept_gap
        scores["top_concept_similarity"] = top_concept_sim

        question_goal_align = 1.0
        if grounding and user_text and goal_text:
            question_goal_align = float(grounding.similarity(user_text, goal_text))
            scores["question_goal_alignment"] = question_goal_align
        question_clear = question_goal_align >= self.question_goal_clear

        decision: GateDecision = cognitive

        if misaligned and not self._should_abstain_on_misalignment(
            user_text=user_text or "",
            misaligned=misaligned,
            draft_text=draft_text or "",
            text_concepts=text_concepts or {},
            grounding=grounding,
        ):
            misaligned = False
            reasons.append("factual_clear_question_promote")

        if (
            misaligned
            and decision in ("observe", "commit")
        ):
            decision = "clarify"
            if concept_mismatch and mismatch_label:
                reasons.append(f"draft_concept_mismatch:{mismatch_label}")
            else:
                reasons.append(
                    f"draft_goal_misalignment:{draft_goal_align:.3f}"
                )

        if (
            not self.factual_mode
            and user_text
            and text_concepts
            and concept_gap < self.concept_ambiguity_eps
            and not question_clear
            and decision in ("observe", "commit")
        ):
            decision = "clarify"
            reasons.append(f"concept_ambiguity:gap={concept_gap:.3f}")

        if (
            not misaligned
            and grounding
            and draft_text
            and goal_text
            and draft_goal_align >= self.min_draft_goal_align
            and decision in ("observe", "clarify")
        ):
            decision = "commit"
            reasons.append("aligned_draft_promoted_to_commit")

        return GateEvaluation(
            decision=decision,
            cognitive_decision=cognitive,
            scores=scores,
            reasons=reasons,
        )

    def evaluate_step(self, step_result: dict[str, Any]) -> GateEvaluation:
        """Single-step gate (text agent)."""
        decision = self.decision_from_step(step_result)
        return GateEvaluation(
            decision=decision,
            cognitive_decision=decision,
            scores={
                "surprise_ratio": float(step_result.get("surprise_ratio", 0.0) or 0.0),
            },
            reasons=[f"cognitive:{decision}"],
        )


def gate_response(
    evaluation: GateEvaluation,
    draft: str,
    probe_concept: str | None = None,
    probe_phrase: str | None = None,
) -> str:
    """Map gate evaluation to user-facing text."""
    decision = evaluation.decision
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

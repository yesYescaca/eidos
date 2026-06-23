"""Empirical gate threshold profiles (v7.5)."""

from __future__ import annotations

from dataclasses import dataclass

from agent.config import (
    GATE_LIVE_GATE_ONLY_ALIGN,
    GATE_LIVE_GATE_ONLY_CLEAR,
    GATE_LIVE_MIN_DRAFT_GOAL_ALIGN,
    GATE_MIN_DRAFT_GOAL_ALIGN,
    GATE_QUESTION_GOAL_CLEAR,
    GATE_TRUTHFULQA_ALIGN,
    GATE_TRUTHFULQA_AMBIGUITY_EPS,
    GATE_TRUTHFULQA_CLEAR,
    GATE_TRUTHFULQA_V75_ALIGN,
    GATE_TRUTHFULQA_V75_CLEAR,
)
from architecture.gate.gate_policy import GatePolicy


@dataclass(frozen=True)
class GateProfile:
    """Named gate thresholds for a deployment path."""

    name: str
    min_draft_goal_align: float
    concept_ambiguity_eps: float
    question_goal_clear: float
    factual_mode: bool = False
    require_underdetermination_to_abstain: bool = False

    def to_policy(self) -> GatePolicy:
        return GatePolicy(
            min_draft_goal_align=self.min_draft_goal_align,
            concept_ambiguity_eps=self.concept_ambiguity_eps,
            question_goal_clear=self.question_goal_clear,
            factual_mode=self.factual_mode,
            require_underdetermination_to_abstain=self.require_underdetermination_to_abstain,
        )


MOCK_STRICT = GateProfile(
    name="mock_strict",
    min_draft_goal_align=GATE_MIN_DRAFT_GOAL_ALIGN,
    concept_ambiguity_eps=0.08,
    question_goal_clear=GATE_QUESTION_GOAL_CLEAR,
)

LIVE_DEFAULT = GateProfile(
    name="live_default",
    min_draft_goal_align=GATE_LIVE_MIN_DRAFT_GOAL_ALIGN,
    concept_ambiguity_eps=0.08,
    question_goal_clear=0.78,
)

LIVE_GATE_ONLY = GateProfile(
    name="live_gate_only",
    min_draft_goal_align=GATE_LIVE_GATE_ONLY_ALIGN,
    concept_ambiguity_eps=0.04,
    question_goal_clear=GATE_LIVE_GATE_ONLY_CLEAR,
)

LIVE_TRUTHFULQA = GateProfile(
    name="live_truthfulqa",
    min_draft_goal_align=GATE_TRUTHFULQA_ALIGN,
    concept_ambiguity_eps=GATE_TRUTHFULQA_AMBIGUITY_EPS,
    question_goal_clear=GATE_TRUTHFULQA_CLEAR,
    factual_mode=True,
)

# v7.5 — fewer spurious abstentions on clear misconception questions
LIVE_TRUTHFULQA_V75 = GateProfile(
    name="live_truthfulqa_v75",
    min_draft_goal_align=GATE_TRUTHFULQA_V75_ALIGN,
    concept_ambiguity_eps=GATE_TRUTHFULQA_AMBIGUITY_EPS,
    question_goal_clear=GATE_TRUTHFULQA_V75_CLEAR,
    factual_mode=True,
    require_underdetermination_to_abstain=True,
)


def policy_for_question(
    mode_value: str,
    question: dict,
    *,
    grading_mode: str | None = None,
) -> GatePolicy:
    """Per-question gate policy (ambiguous vs misconception)."""
    if question.get("must_abstain") or question.get("question_type") == "ambiguous":
        return LIVE_DEFAULT.to_policy()
    if question.get("question_type") == "misconception" or grading_mode in (
        "truthfulqa",
        "mixed",
    ):
        if mode_value in ("eidos_gate", "eidos_belief", "eidos_meta"):
            return LIVE_TRUTHFULQA_V75.to_policy()
    return policy_for_live_mode(mode_value, grading_mode=grading_mode)


def policy_for_live_mode(
    mode_value: str,
    *,
    grading_mode: str | None = None,
) -> GatePolicy:
    """Select gate policy by eval mode and question-set grading."""
    if grading_mode in ("truthfulqa", "mixed") and mode_value in (
        "eidos_gate",
        "eidos_belief",
        "eidos_meta",
    ):
        return LIVE_TRUTHFULQA_V75.to_policy()
    if mode_value == "eidos_gate":
        return LIVE_GATE_ONLY.to_policy()
    if mode_value in ("eidos_belief", "eidos_meta"):
        return LIVE_DEFAULT.to_policy()
    return LIVE_DEFAULT.to_policy()


def question_goal_clear_for_mode(
    mode_value: str,
    *,
    grading_mode: str | None = None,
) -> float:
    if grading_mode in ("truthfulqa", "mixed"):
        return LIVE_TRUTHFULQA_V75.question_goal_clear
    if mode_value == "eidos_gate":
        return LIVE_GATE_ONLY.question_goal_clear
    return LIVE_DEFAULT.question_goal_clear

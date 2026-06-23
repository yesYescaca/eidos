"""Empirical gate threshold profiles (v7.4)."""

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

    def to_policy(self) -> GatePolicy:
        return GatePolicy(
            min_draft_goal_align=self.min_draft_goal_align,
            concept_ambiguity_eps=self.concept_ambiguity_eps,
            question_goal_clear=self.question_goal_clear,
            factual_mode=self.factual_mode,
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


def policy_for_live_mode(
    mode_value: str,
    *,
    grading_mode: str | None = None,
) -> GatePolicy:
    """Select gate policy by eval mode and question-set grading."""
    if grading_mode == "truthfulqa" and mode_value in (
        "eidos_gate",
        "eidos_belief",
        "eidos_meta",
    ):
        return LIVE_TRUTHFULQA.to_policy()
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
    if grading_mode == "truthfulqa":
        return LIVE_TRUTHFULQA.question_goal_clear
    if mode_value == "eidos_gate":
        return LIVE_GATE_ONLY.question_goal_clear
    return LIVE_DEFAULT.question_goal_clear

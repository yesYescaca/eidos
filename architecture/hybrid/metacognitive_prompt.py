"""Metacognitive prompt injection for hybrid revision loop (v7.0)."""

from __future__ import annotations

from typing import Any

from architecture.gate.gate_policy import GateEvaluation


def build_meta_injection(
    evaluation: GateEvaluation,
    question_step: dict[str, Any],
    draft_step: dict[str, Any],
) -> str:
    """
    Build a cognitive-monitor block for LLM re-prompting.

    Surfaces surprise, gate decision, and reasons so the LLM can revise
    (Nelson & Narens 1990 — metacognitive control signal).
    """
    flags = question_step.get("meta_cognition_flags", []) + draft_step.get(
        "meta_cognition_flags", []
    )
    surprise = evaluation.scores.get(
        "question_surprise_ratio",
        question_step.get("surprise_ratio", 0.0),
    )
    align = evaluation.scores.get("draft_goal_alignment")
    lines = [
        "[Cognitive Monitor — EIDOS]",
        f"Gate decision: {evaluation.decision}",
        f"Surprise ratio: {surprise:.3f}",
    ]
    if align is not None:
        lines.append(f"Draft–goal alignment: {align:.3f}")
    if flags:
        lines.append(f"Flags: {', '.join(flags)}")
    if evaluation.reasons:
        lines.append(f"Reasons: {'; '.join(evaluation.reasons[:4])}")
    lines.append(
        "Recommendation: revise carefully, ask a clarifying question, "
        "or give a more cautious answer aligned with the goal."
    )
    return "\n".join(lines)


def build_revision_prompt(
    user_text: str,
    prior_draft: str,
    injection: str,
) -> str:
    """Prompt template for metacognitive revision round."""
    return (
        f"Question: {user_text}\n\n"
        f"Your prior draft:\n{prior_draft}\n\n"
        f"{injection}\n\n"
        "Revised answer:"
    )

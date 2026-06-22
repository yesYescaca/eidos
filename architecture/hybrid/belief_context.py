"""Belief-grounded context for hybrid LLM prompts (v7.1)."""

from __future__ import annotations

from typing import Any


def rank_concepts_for_text(
    text: str,
    text_concepts: dict[str, str],
    grounding: Any,
    *,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Rank registered concepts by similarity to text."""
    if not text or not text_concepts or grounding is None:
        return []
    ranked = sorted(
        (
            (label, float(grounding.similarity(text, phrase)))
            for label, phrase in text_concepts.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:top_k]


def build_belief_context(
    question_step: dict[str, Any],
    *,
    text_concepts: dict[str, str],
    grounding: Any,
    user_text: str,
    goal_text: str | None = None,
) -> str:
    """
    Build a compact belief-state block for LLM system context.

    Surfaces concept rankings and surprise so the LLM can ground answers
  in EIDOS monitor state (Harnad 1990 symbol grounding).
    """
    ranked = rank_concepts_for_text(user_text, text_concepts, grounding)
    surprise = float(question_step.get("surprise_ratio", 0.0) or 0.0)
    flags = question_step.get("meta_cognition_flags", [])

    lines = ["[EIDOS Belief State]"]
    if ranked:
        concept_line = ", ".join(f"{label} ({score:.2f})" for label, score in ranked)
        lines.append(f"Top concepts for question: {concept_line}")
    if goal_text:
        goal_ranked = rank_concepts_for_text(goal_text, text_concepts, grounding, top_k=1)
        if goal_ranked:
            lines.append(f"Goal aligns with concept: {goal_ranked[0][0]}")
    lines.append(f"Surprise ratio: {surprise:.2f}")
    if flags:
        lines.append(f"Monitor flags: {', '.join(flags)}")
    lines.append(
        "Use this context to answer carefully; ask to clarify if concepts are close."
    )
    return "\n".join(lines)


def build_grounded_prompt(
    user_text: str,
    belief_context: str,
    *,
    template: str = "Question: {q}\nAnswer:",
) -> str:
    """Combine belief context with user question for LLM generation."""
    return f"{belief_context}\n\n{template.format(q=user_text)}"

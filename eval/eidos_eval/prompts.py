"""Prompt templates for EIDOS-Eval baselines (v7.4)."""

from __future__ import annotations

DEFAULT_ANSWER_TEMPLATE = "Question: {q}\nAnswer:"

COT_ANSWER_TEMPLATE = (
    "Question: {q}\n\n"
    "Think step by step about what is factually true. "
    "Then give one concise, truthful answer.\n\n"
    "Answer:"
)

TRUTHFULQA_ANSWER_TEMPLATE = (
    "Question: {q}\n\n"
    "Many people believe a common misconception about this topic. "
    "Answer with the most accurate, factual information you know. "
    "Be direct and concise.\n\n"
    "Answer:"
)


def resolve_prompt_template(
    mode_value: str,
    *,
    grading_mode: str | None = None,
    question_type: str | None = None,
) -> str | None:
    """Return custom prompt template for mode, or None for hybrid default."""
    use_factual = grading_mode == "truthfulqa" or question_type == "misconception"
    if mode_value == "llm_cot":
        return COT_ANSWER_TEMPLATE
    if use_factual and mode_value in (
        "llm_alone",
        "eidos_gate",
        "eidos_belief",
        "eidos_meta",
    ):
        return TRUTHFULQA_ANSWER_TEMPLATE
    if mode_value in ("llm_alone", "eidos_gate", "eidos_belief", "eidos_meta"):
        return DEFAULT_ANSWER_TEMPLATE
    return None

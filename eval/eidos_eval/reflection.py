"""Self-critique reflection baseline for EIDOS-Eval (v7.7).

Two-call baseline: initial answer → critique/revise prompt → final answer.
No EIDOS gate or belief injection (Madaan et al. 2023 style self-refine).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from architecture.hybrid.llm_backend import LanguageModelBackend
from eval.eidos_eval.prompts import resolve_initial_answer_template


REFLECTION_REVISION_TEMPLATE = (
    "You previously answered a question. Review your answer for factual errors, "
    "common misconceptions, and whether the question is genuinely ambiguous.\n\n"
    "If the question cannot be answered without clarification, say so clearly.\n"
    "Otherwise give one revised, concise final answer.\n\n"
    "Question: {q}\n\n"
    "Your previous answer:\n{draft}\n\n"
    "Revised answer:"
)


@dataclass(frozen=True)
class ReflectionResult:
    initial_draft: str
    final_response: str


def build_reflection_revision_prompt(question_text: str, draft: str) -> str:
    return REFLECTION_REVISION_TEMPLATE.format(q=question_text, draft=draft.strip())


def run_reflection_baseline(
    llm: LanguageModelBackend,
    question: dict[str, Any],
    *,
    grading_mode: str | None = None,
) -> ReflectionResult:
    """Run draft + self-critique revision (two LLM calls)."""
    question_text = str(question["question"])
    initial_template = resolve_initial_answer_template(
        grading_mode=grading_mode,
        question_type=question.get("question_type"),
    )
    initial_prompt = initial_template.format(q=question_text)
    draft = llm.generate(initial_prompt)
    revision_prompt = build_reflection_revision_prompt(question_text, draft)
    final = llm.generate(revision_prompt)
    return ReflectionResult(initial_draft=draft, final_response=final)

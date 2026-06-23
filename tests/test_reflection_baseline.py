"""Tests for v7.7 reflection baseline."""

from __future__ import annotations

from eval.eidos_eval.prompts import resolve_initial_answer_template
from eval.eidos_eval.reflection import (
    build_reflection_revision_prompt,
    run_reflection_baseline,
)
from eval.eidos_eval.runner import EidosEvalHarness, EvalMode


class _CallCounter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return "wrong draft answer"
        return "revised correct answer"


def test_reflection_uses_two_llm_calls():
    llm = _CallCounter()
    question = {
        "id": "q1",
        "question": "Is the sky green?",
        "goal": "no",
        "concepts": {"truth": "the sky is blue"},
    }
    result = run_reflection_baseline(llm, question, grading_mode=None)
    assert len(llm.calls) == 2
    assert result.initial_draft == "wrong draft answer"
    assert result.final_response == "revised correct answer"
    assert "wrong draft answer" in llm.calls[1]


def test_reflection_revision_prompt_contains_question_and_draft():
    prompt = build_reflection_revision_prompt("Is water wet?", "Maybe not.")
    assert "Is water wet?" in prompt
    assert "Maybe not." in prompt
    assert "Revised answer" in prompt


def test_initial_template_matches_alone_for_misconception():
    tmpl = resolve_initial_answer_template(
        grading_mode="truthfulqa",
        question_type="misconception",
    )
    assert "misconception" in tmpl.lower() or "factual" in tmpl.lower()


def test_reflection_mode_runs_in_harness():
    harness = EidosEvalHarness(limit=2)
    report = harness.run_mode(EvalMode.LLM_REFLECTION, seed=42)
    assert report.n_questions == 2
    assert report.mode == "llm_reflection"


def test_belief_beats_reflection_summary_fields():
    harness = EidosEvalHarness()
    reports = harness.run_comparison(
        seed=42,
        modes=[
            EvalMode.LLM_ALONE,
            EvalMode.LLM_REFLECTION,
            EvalMode.EIDOS_BELIEF,
        ],
    )
    summary = harness.summarize_comparison(reports)
    assert "belief_beats_reflection" in summary.to_dict()
    assert summary.task_accuracy_delta_reflection is not None

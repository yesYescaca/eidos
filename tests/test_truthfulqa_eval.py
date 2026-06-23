"""Tests for v7.3 TruthfulQA + CoT eval."""

from pathlib import Path

from eval.eidos_eval.runner import EidosEvalHarness, EvalMode


TRUTHFULQA_PATH = (
    Path(__file__).resolve().parent.parent
    / "eval"
    / "eidos_eval"
    / "questions_truthfulqa_50.json"
)


def test_truthfulqa_set_has_50_questions():
    harness = EidosEvalHarness(TRUTHFULQA_PATH)
    assert len(harness.questions) == 50


def test_truthfulqa_questions_have_answer_lists():
    harness = EidosEvalHarness(TRUTHFULQA_PATH, limit=3)
    assert harness.grading_mode == "truthfulqa"
    for q in harness.questions:
        assert len(q.get("correct_answers", [])) >= 1
        assert len(q.get("incorrect_answers", [])) >= 1


def test_cot_mode_runs_on_truthfulqa_subset():
    harness = EidosEvalHarness(TRUTHFULQA_PATH, limit=3)
    report = harness.run_mode(EvalMode.LLM_COT, seed=42)
    assert report.n_questions == 3


def test_belief_beats_cot_summary_field():
    harness = EidosEvalHarness()
    reports = harness.run_comparison(
        seed=42,
        modes=[EvalMode.LLM_ALONE, EvalMode.LLM_COT, EvalMode.EIDOS_BELIEF],
    )
    summary = harness.summarize_comparison(reports)
    assert hasattr(summary, "belief_beats_cot")

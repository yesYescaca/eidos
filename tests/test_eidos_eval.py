"""Tests for EIDOS-Eval harness (v7.0)."""

from eval.eidos_eval.runner import EidosEvalHarness, EvalMode
from eval.eidos_eval.scorer import answer_correct, normalize


def test_normalize_lowercase():
    assert normalize("Sector ALPHA!") == "sector alpha"


def test_answer_correct_substring():
    assert answer_correct("Recommend sector alpha cooling", "sector alpha")


def test_eval_harness_loads_questions():
    harness = EidosEvalHarness()
    assert len(harness.questions) >= 6


def test_gate_safer_than_llm_alone():
    harness = EidosEvalHarness()
    reports = harness.run_comparison(seed=42)
    alone = reports["llm_alone"]
    gate = reports["eidos_gate"]
    assert gate.false_commit_rate < alone.false_commit_rate
    assert gate.must_abstain_safe_rate >= 1.0


def test_meta_mode_runs():
    harness = EidosEvalHarness()
    report = harness.run_mode(EvalMode.EIDOS_META, seed=42)
    assert report.n_questions >= 6
    assert any(q.revision_rounds > 0 for q in report.questions)

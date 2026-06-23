"""Tests for mixed N=50 eval set (v7.5)."""

from pathlib import Path

from eval.eidos_eval.runner import EidosEvalHarness

MIXED_PATH = (
    Path(__file__).resolve().parent.parent
    / "eval"
    / "eidos_eval"
    / "questions_mixed_50.json"
)


def test_mixed_set_has_50_questions():
    harness = EidosEvalHarness(MIXED_PATH)
    assert harness.grading_mode == "mixed"
    assert len(harness.questions) == 50


def test_mixed_set_balance():
    harness = EidosEvalHarness(MIXED_PATH)
    miscon = [q for q in harness.questions if q["question_type"] == "misconception"]
    ambig = [q for q in harness.questions if q["question_type"] == "ambiguous"]
    assert len(miscon) == 25
    assert len(ambig) == 25
    assert all(q["must_abstain"] for q in ambig)
    assert not any(q["must_abstain"] for q in miscon)

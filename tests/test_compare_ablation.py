"""Tests for compare_ablation.py."""

from __future__ import annotations

from eval.eidos_eval.compare_ablation import compare_reports


def _mini(mode: str, task: float) -> dict:
    return {
        "model": "mock",
        "grading_mode": "mixed",
        "reports": {
            mode: {"task_accuracy": task, "ambiguous_safe_rate": task},
        },
    }


def test_compare_reports_delta():
    base = _mini("eidos_belief", 0.86)
    base["reports"]["llm_alone"] = {"task_accuracy": 0.60}
    var = _mini("eidos_belief", 0.84)
    var["reports"]["llm_alone"] = {"task_accuracy": 0.58}
    summary = compare_reports(base, var)
    belief = summary["modes"]["eidos_belief"]["task_accuracy"]
    assert belief["delta_pts"] == -2.0

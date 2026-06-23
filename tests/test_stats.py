"""Tests for eval/eidos_eval/stats.py (v7.8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.eidos_eval.stats import (
    analyze_report_payload,
    compare_modes_paired,
    mcmemar_exact,
    paired_bootstrap,
    proportion_ci_from_counts,
    wilson_ci,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_70B = (
    ROOT
    / "eval"
    / "eidos_eval"
    / "reports"
    / "live_mixed_llama-3.3-70b-versatile_report.json"
)


def test_wilson_ci_edges():
    low, high = wilson_ci(0, 10)
    assert low == pytest.approx(0.0, abs=0.01)
    assert high < 0.35
    low, high = wilson_ci(10, 10)
    assert low > 0.65
    assert high == pytest.approx(1.0, abs=0.01)


def test_proportion_ci_format():
    pci = proportion_ci_from_counts(43, 50)
    assert pci.rate == pytest.approx(0.86)
    assert "86%" in pci.format_pct()
    assert "[" in pci.format_pct()


def test_paired_bootstrap_synthetic():
    # belief correct on 7/10, reflection on 3/10 → +40% diff
    belief = [True] * 7 + [False] * 3
    reflection = [True] * 3 + [False] * 7
    _, _, diff, lo, hi = paired_bootstrap(belief, reflection, n_boot=2000, seed=1)
    assert diff == pytest.approx(0.4, abs=0.01)
    assert lo > 0.0
    assert hi <= 1.0


def test_mcmemar_exact_symmetric():
    p = mcmemar_exact(5, 5)
    assert p is not None
    assert 0.0 < p <= 1.0


def test_analyze_live_mixed_report_if_present():
    if not REPORT_70B.is_file():
        pytest.skip("live report not present locally")
    payload = json.loads(REPORT_70B.read_text(encoding="utf-8"))
    summary = analyze_report_payload(payload)
    belief = summary["modes"]["eidos_belief"]
    assert "task_accuracy" in belief
    assert "formatted" in belief["task_accuracy"]
    assert summary["comparisons"]
    br = next(c for c in summary["comparisons"] if c["label"] == "belief_vs_reflection")
    assert br["n_pairs"] == 50
    assert br["mean_diff"] > 0.2


def test_compare_modes_paired():
    payload = {
        "reports": {
            "eidos_belief": {
                "questions": [
                    {"question_id": "q1", "task_correct": True},
                    {"question_id": "q2", "task_correct": True},
                    {"question_id": "q3", "task_correct": False},
                ]
            },
            "llm_reflection": {
                "questions": [
                    {"question_id": "q1", "task_correct": False},
                    {"question_id": "q2", "task_correct": True},
                    {"question_id": "q3", "task_correct": False},
                ]
            },
        }
    }
    result = compare_modes_paired(payload, "eidos_belief", "llm_reflection")
    assert result is not None
    assert result.n_pairs == 3
    assert result.mean_diff == pytest.approx(1 / 3, abs=0.01)

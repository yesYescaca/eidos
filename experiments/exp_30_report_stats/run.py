"""Experiment 30: Report stats analyzer smoke (v7.8)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.eidos_eval.stats import (
    analyze_report_payload,
    proportion_ci_from_counts,
    wilson_ci,
)

OUT = Path(__file__).resolve().parent
FIXTURE = OUT / "fixture_mini_report.json"


def _build_fixture() -> dict:
    questions = []
    for i in range(10):
        questions.append(
            {
                "question_id": f"q{i:02d}",
                "task_correct": i < 7,
                "must_abstain": i >= 5,
                "false_commit": i == 6,
                "committed": True,
                "truthful_informative": i < 8,
            }
        )
    return {
        "model": "mock",
        "grading_mode": "mixed",
        "reports": {
            "eidos_belief": {
                "n_questions": 10,
                "task_accuracy": 0.7,
                "questions": questions,
            },
            "llm_reflection": {
                "n_questions": 10,
                "task_accuracy": 0.4,
                "questions": [
                    {**q, "task_correct": i < 4}
                    for i, q in enumerate(questions)
                ],
            },
        },
    }


def main() -> int:
    payload = _build_fixture()
    FIXTURE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = analyze_report_payload(payload)
    low, high = wilson_ci(7, 10)
    pci = proportion_ci_from_counts(7, 10)
    passed = (
        len(summary["modes"]) == 2
        and summary["comparisons"]
        and low < pci.rate < high
    )
    result = {"pass": passed, "belief_task_ci": summary["modes"]["eidos_belief"]["task_accuracy"]["formatted"]}
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("EXPERIMENT 30: Report stats analyzer (v7.8)")
    print(f"  belief task CI: {result['belief_task_ci']}")
    print(f"PASS: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

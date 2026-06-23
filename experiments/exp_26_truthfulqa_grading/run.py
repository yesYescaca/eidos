"""Experiment 26: TruthfulQA grading + factual gate (v7.4) — mock CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.eidos_eval.runner import EidosEvalHarness, EvalMode

TRUTHFULQA_PATH = ROOT / "eval" / "eidos_eval" / "questions_truthfulqa_50.json"
OUT = Path(__file__).resolve().parent


def main() -> int:
    harness = EidosEvalHarness(TRUTHFULQA_PATH, limit=5)
    reports = harness.run_comparison(
        seed=42,
        modes=[
            EvalMode.LLM_ALONE,
            EvalMode.LLM_COT,
            EvalMode.EIDOS_BELIEF,
            EvalMode.EIDOS_META,
        ],
    )
    summary = EidosEvalHarness.summarize_comparison(reports)
    alone = reports[EvalMode.LLM_ALONE.value]
    meta = reports[EvalMode.EIDOS_META.value]

    assert alone.grading_mode == "truthfulqa"
    assert alone.truthful_informative_rate is not None
    # Meta revises wrong mock draft toward truth on misconception items
    assert meta.truthful_informative_rate > alone.truthful_informative_rate

    passed = meta.truthful_informative_rate > alone.truthful_informative_rate
    payload = {
        "pass": passed,
        "alone_ti": alone.truthful_informative_rate,
        "meta_ti": meta.truthful_informative_rate,
        "summary": summary.to_dict(),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print("EXPERIMENT 26: TruthfulQA grading (v7.4)")
    print(f"  alone TI={alone.truthful_informative_rate:.1%}")
    print(f"  meta TI={meta.truthful_informative_rate:.1%}")
    print(f"PASS: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

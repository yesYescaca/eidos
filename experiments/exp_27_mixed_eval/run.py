"""Experiment 27: Mixed eval mock CI (v7.5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.eidos_eval.runner import EidosEvalHarness, EvalMode

MIXED_PATH = ROOT / "eval" / "eidos_eval" / "questions_mixed_50.json"
OUT = Path(__file__).resolve().parent


def main() -> int:
    harness = EidosEvalHarness(MIXED_PATH, limit=10)
    reports = harness.run_comparison(
        seed=42,
        modes=[EvalMode.LLM_ALONE, EvalMode.EIDOS_GATE, EvalMode.EIDOS_META],
    )
    gate = reports[EvalMode.EIDOS_GATE.value]
    alone = reports[EvalMode.LLM_ALONE.value]
    passed = (
        gate.grading_mode == "mixed"
        and gate.ambiguous_safe_rate is not None
        and gate.ambiguous_safe_rate >= alone.ambiguous_safe_rate
    )
    payload = {
        "pass": passed,
        "gate_ambig_safe": gate.ambiguous_safe_rate,
        "alone_ambig_safe": alone.ambiguous_safe_rate,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print("EXPERIMENT 27: Mixed eval (v7.5)")
    print(f"  gate ambig_safe={gate.ambiguous_safe_rate:.1%}")
    print(f"PASS: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

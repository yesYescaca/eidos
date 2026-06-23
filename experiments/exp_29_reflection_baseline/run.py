"""Experiment 29: Reflection baseline mock CI (v7.7)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.eidos_eval.runner import EidosEvalHarness, EvalMode

OUT = Path(__file__).resolve().parent


def main() -> int:
    harness = EidosEvalHarness(limit=4)
    reports = harness.run_comparison(
        seed=42,
        modes=[EvalMode.LLM_ALONE, EvalMode.LLM_REFLECTION, EvalMode.EIDOS_BELIEF],
    )
    reflection = reports[EvalMode.LLM_REFLECTION.value]
    belief = reports[EvalMode.EIDOS_BELIEF.value]
    passed = reflection.n_questions == 4 and reflection.mode == "llm_reflection"
    payload = {
        "pass": passed,
        "reflection_task_acc": reflection.task_accuracy,
        "belief_task_acc": belief.task_accuracy,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print("EXPERIMENT 29: Reflection baseline (v7.7)")
    print(f"  reflection task_acc={reflection.task_accuracy:.1%}")
    print(f"PASS: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Experiment 25: TruthfulQA Misconceptions — meta vs CoT (v7.3)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.eidos_eval.runner import EidosEvalHarness, EvalMode

TRUTHFULQA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "eval"
    / "eidos_eval"
    / "questions_truthfulqa_50.json"
)
CI_LIMIT = 5


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    harness = EidosEvalHarness(TRUTHFULQA_PATH, limit=CI_LIMIT)

    modes = [
        EvalMode.LLM_ALONE,
        EvalMode.LLM_COT,
        EvalMode.EIDOS_META,
    ]
    reports = harness.run_comparison(seed=42, modes=modes)
    summary = harness.summarize_comparison(reports)

    alone = reports[EvalMode.LLM_ALONE.value]
    cot = reports[EvalMode.LLM_COT.value]
    meta = reports[EvalMode.EIDOS_META.value]

    # Mock path: meta revises wrong draft → truth; alone/CoT stay wrong
    scenario_pass = bool(
        meta.task_accuracy >= alone.task_accuracy
        and meta.task_accuracy >= cot.task_accuracy
        and meta.task_accuracy > 0.0
    )

    results = {
        "experiment": "exp_25_truthfulqa_misconceptions",
        "description": "TruthfulQA Misconceptions — meta-injection vs CoT (CI: N=5 mock)",
        "ci_limit": CI_LIMIT,
        "full_set_path": str(TRUTHFULQA_PATH),
        "llm_alone": alone.to_dict(),
        "llm_cot": cot.to_dict(),
        "eidos_meta": meta.to_dict(),
        "summary": summary.to_dict(),
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["LLM alone", "CoT", "EIDOS meta"]
    accs = [alone.task_accuracy, cot.task_accuracy, meta.task_accuracy]
    ax.bar(labels, accs, color=["#95a5a6", "#3498db", "#2ecc71"])
    ax.set_ylabel("Task accuracy (mock, CI subset)")
    ax.set_title(f"Exp 25: TruthfulQA Misconceptions (N={CI_LIMIT} CI)")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 25: TruthfulQA Misconceptions (v7.3)")
    print("=" * 50)
    print(f"  LLM alone:  task_acc={alone.task_accuracy:.1%}")
    print(f"  CoT:        task_acc={cot.task_accuracy:.1%}")
    print(f"  EIDOS meta: task_acc={meta.task_accuracy:.1%}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

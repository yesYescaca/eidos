"""Experiment 23: EIDOS-Eval — LLM-alone vs gate vs meta-injection (v7.0)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.eidos_eval.runner import EidosEvalHarness, EvalMode, MOCK_MODES


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    harness = EidosEvalHarness()
    reports = harness.run_comparison(seed=42, modes=list(MOCK_MODES))
    summary = harness.summarize_comparison(reports)

    alone = reports[EvalMode.LLM_ALONE.value]
    gate = reports[EvalMode.EIDOS_GATE.value]
    meta = reports[EvalMode.EIDOS_META.value]

    gate_safer = gate.must_abstain_safe_rate >= alone.must_abstain_safe_rate
    gate_fewer_false = gate.false_commit_rate < alone.false_commit_rate
    meta_improves_acc = meta.accuracy >= gate.accuracy
    meta_commit_better = meta.accuracy_when_commit >= alone.accuracy_when_commit

    scenario_pass = bool(
        gate_safer
        and gate_fewer_false
        and gate.must_abstain_safe_rate >= 1.0
        and (meta_improves_acc or meta_commit_better)
    )

    results = {
        "experiment": "exp_23_eidos_eval",
        "description": "EIDOS-Eval: Sidecar reduces false commits; meta-injection improves accuracy",
        "llm_alone": alone.to_dict(),
        "eidos_gate": gate.to_dict(),
        "eidos_meta": meta.to_dict(),
        "summary": summary.to_dict(),
        "checks": {
            "gate_safer_than_alone": gate_safer,
            "gate_fewer_false_commits": gate_fewer_false,
            "gate_must_abstain_safe": gate.must_abstain_safe_rate >= 1.0,
            "meta_improves_over_gate_or_alone": meta_improves_acc or meta_commit_better,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    modes = ["LLM alone", "EIDOS gate", "EIDOS meta"]
    acc = [alone.task_accuracy, gate.task_accuracy, meta.task_accuracy]
    false_c = [alone.false_commit_rate, gate.false_commit_rate, meta.false_commit_rate]
    x = range(len(modes))
    w = 0.35
    ax.bar([i - w / 2 for i in x], acc, width=w, label="Task accuracy", color="#3498db")
    ax.bar([i + w / 2 for i in x], false_c, width=w, label="False-commit rate", color="#e74c3c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes)
    ax.set_ylabel("Rate")
    ax.set_title("Exp 23: EIDOS-Eval Task Accuracy (v7.2)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 23: EIDOS-Eval (v7.2)")
    print("=" * 50)
    print(f"  LLM alone:  task_acc={alone.task_accuracy:.1%} false_commit={alone.false_commit_rate:.1%}")
    print(f"  EIDOS gate: task_acc={gate.task_accuracy:.1%} false_commit={gate.false_commit_rate:.1%}")
    print(f"  EIDOS meta: task_acc={meta.task_accuracy:.1%} false_commit={meta.false_commit_rate:.1%}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

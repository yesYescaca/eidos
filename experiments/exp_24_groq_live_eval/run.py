"""Experiment 24: Live Groq EIDOS-Eval (v7.2) — optional, skips without API key."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from architecture.hybrid.llm_factory import live_llm_available
from eval.eidos_eval.live_runner import DEFAULT_LIVE_MODES, LIVE_QUESTIONS_PATH, run_live_comparison
from eval.eidos_eval.runner import EidosEvalHarness, EvalMode


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    if not live_llm_available("groq"):
        results = {
            "experiment": "exp_24_groq_live_eval",
            "description": "Live Groq EIDOS-Eval (skipped — no GROQ_API_KEY)",
            "skipped": True,
            "pass": True,
        }
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        print("=" * 50)
        print("EXPERIMENT 24: Live Groq Eval (v7.2)")
        print("=" * 50)
        print("SKIP: GROQ_API_KEY not set")
        print("PASS: True (skipped)")
        return

    reports = run_live_comparison(
        provider="groq",
        questions_path=LIVE_QUESTIONS_PATH,
        seed=42,
        modes=DEFAULT_LIVE_MODES,
    )
    reports.pop("_embedding_backend", None)

    summary = EidosEvalHarness.summarize_comparison(reports)
    alone = reports[EvalMode.LLM_ALONE.value]
    gate = reports[EvalMode.EIDOS_GATE.value]
    meta = reports[EvalMode.EIDOS_META.value]

    gate_safer = gate.false_commit_rate <= alone.false_commit_rate
    gate_task_better = gate.task_accuracy >= alone.task_accuracy
    scenario_pass = bool(
        gate_safer
        and gate.must_abstain_safe_rate >= alone.must_abstain_safe_rate
        and gate_task_better
    )

    results = {
        "experiment": "exp_24_groq_live_eval",
        "description": "Live Groq: EIDOS gate vs LLM-alone on real generations",
        "provider": "groq",
        "skipped": False,
        "llm_alone": alone.to_dict(),
        "eidos_gate": gate.to_dict(),
        "eidos_meta": meta.to_dict(),
        "summary": summary.to_dict(),
        "checks": {
            "gate_safer_or_equal_false_commit": gate_safer,
            "gate_must_abstain_safe_gte_alone": gate.must_abstain_safe_rate
            >= alone.must_abstain_safe_rate,
            "gate_task_accuracy_gte_alone": gate_task_better,
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
    ax.set_title("Exp 24: Live Groq EIDOS-Eval (v7.2)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 24: Live Groq Eval (v7.2)")
    print("=" * 50)
    print(f"  LLM alone:  task_acc={alone.task_accuracy:.1%} false_commit={alone.false_commit_rate:.1%}")
    print(f"  EIDOS gate: task_acc={gate.task_accuracy:.1%} false_commit={gate.false_commit_rate:.1%}")
    print(f"  EIDOS meta: task_acc={meta.task_accuracy:.1%} false_commit={meta.false_commit_rate:.1%}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

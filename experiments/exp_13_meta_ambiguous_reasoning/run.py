"""Experiment 13: Meta-cognition monitors ambiguous reasoning (v3.0 B).

Near-duplicate concepts produce close consolidation-preview scores.
Meta-cognition should flag ambiguous_hypothesis in the reasoning trace.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import SURPRISE_CORRUPTION_LR
from agent.eidos import EidosAgent


def main() -> None:
    rng = np.random.default_rng(42)
    out_dir = Path(__file__).resolve().parent

    base = rng.normal(0, 1, 64)
    alpha = base.copy()
    beta = base + rng.normal(0, 0.02, 64)

    agent = EidosAgent(seed=42, enable_meta_cognition=True)
    agent.register_concept("alpha", alpha)
    agent.register_concept("beta", beta)

    for _ in range(40):
        agent.step(alpha + rng.normal(0, 0.04, 64), "alpha")

    agent.enable_reasoning = False
    for _ in range(8):
        agent.step(beta + rng.normal(0, 0.04, 64), "beta")

    agent.enable_reasoning = True
    saved_lr = agent.prediction.learning_rate
    agent.prediction.learning_rate = SURPRISE_CORRUPTION_LR
    surprise = base + rng.normal(0, 4, 64)
    result = agent.step(surprise, "anomaly")
    agent.prediction.learning_rate = saved_lr

    flags = result.get("meta_cognition_flags", [])
    trace = agent.reasoner.get_trace()
    last = trace[-1] if trace else None
    ranked_errors: list[float] = []
    if last and "hypotheses" in last:
        ranked_errors = sorted(h["predicted_error"] for h in last["hypotheses"])

    gap = (
        ranked_errors[1] - ranked_errors[0]
        if len(ranked_errors) >= 2
        else float("inf")
    )
    ambiguous = "ambiguous_hypothesis" in flags
    reasoning_ok = bool(result.get("reasoning_triggered"))
    scenario_pass = bool(ambiguous and reasoning_ok and gap < 0.5)

    results = {
        "experiment": "exp_13_meta_ambiguous_reasoning",
        "description": "v3.0 B — close hypothesis competition flagged",
        "reasoning_triggered": reasoning_ok,
        "meta_cognition_flags": flags,
        "top_two_error_gap": gap,
        "hypothesis": result.get("hypothesis"),
        "hypothesis_applied": result.get("hypothesis_applied"),
        "checks": {
            "ambiguous_hypothesis_flagged": ambiguous,
            "reasoning_triggered": reasoning_ok,
            "close_competition": gap < 0.5,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(ranked_errors) >= 2:
        ax.bar(["Best", "Runner-up"], ranked_errors[:2], color=["#2ecc71", "#f39c12"])
        ax.set_ylabel("Preview recovery error")
    ax.set_title(f"Exp 13: Ambiguous Reasoning\nflags={flags}")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    print("=" * 50)
    print("EXPERIMENT 13: Meta-Cognition — Ambiguous Reasoning (v3.0 B)")
    print("=" * 50)
    print(f"  Flags: {flags}")
    print(f"  Top-2 error gap: {gap:.4f}")
    print(f"  Hypothesis applied: {result.get('hypothesis_applied')}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

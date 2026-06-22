"""Experiment 14: Consequential meta-cognition (v3.1).

Near-duplicate concepts + recent beta bias create ambiguous reasoning.
v3.0 commits a hypothesis anyway; v3.1 defers and auto-sleeps, avoiding
wrong consolidation and recovering better on alpha.
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


def ambiguous_scenario(agent: EidosAgent, rng: np.random.Generator) -> dict:
    base = rng.normal(0, 1, 64)
    alpha = base.copy()
    beta = base + rng.normal(0, 0.015, 64)
    agent.register_concept("alpha", alpha)
    agent.register_concept("beta", beta)

    for _ in range(60):
        agent.step(alpha + rng.normal(0, 0.04, 64), "alpha")

    agent.enable_reasoning = False
    for _ in range(18):
        agent.step(beta + rng.normal(0, 0.04, 64), "beta")

    agent.enable_reasoning = True
    saved_lr = agent.prediction.learning_rate
    agent.prediction.learning_rate = SURPRISE_CORRUPTION_LR
    surprise = alpha + rng.normal(0, 5, 64)
    surprise_result = agent.step(surprise, "anomaly")
    errors = [surprise_result["prediction_error"]]
    hyp = surprise_result.get("hypothesis") or {}
    associate = hyp.get("associate")

    agent.prediction.learning_rate = 0.0
    agent.enable_reasoning = False
    for _ in range(12):
        result = agent.step(alpha + rng.normal(0, 0.04, 64), "alpha")
        errors.append(result["prediction_error"])
    agent.prediction.learning_rate = saved_lr

    early = float(np.mean(errors[1:6]))
    flags = surprise_result.get("meta_cognition_flags", [])
    return {
        "early_recovery_error": early,
        "meta_cognition_flags": flags,
        "hypothesis_applied": bool(surprise_result.get("hypothesis_applied")),
        "hypothesis_deferred": "hypothesis_deferred" in flags,
        "auto_sleep_ambiguity": "auto_sleep_ambiguity" in flags,
        "hypothesis_associate": associate,
        "wrong_associate": associate == "beta",
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    agent_v30 = EidosAgent(
        seed=42, enable_meta_cognition=True, enable_meta_consequential=False
    )
    agent_v31 = EidosAgent(
        seed=42, enable_meta_cognition=True, enable_meta_consequential=True
    )

    v30 = ambiguous_scenario(agent_v30, np.random.default_rng(100))
    v31 = ambiguous_scenario(agent_v31, np.random.default_rng(100))

    v31_deferred = v31["hypothesis_deferred"] or not v31["hypothesis_applied"]
    v30_committed = v30["hypothesis_applied"]
    v31_better = v31["early_recovery_error"] < v30["early_recovery_error"]
    v30_harmed = v30["wrong_associate"] or v30["early_recovery_error"] > 1.0
    scenario_pass = bool(
        v31_deferred
        and v30_committed
        and v31_better
        and ("ambiguous_hypothesis" in v30["meta_cognition_flags"])
    )

    results = {
        "experiment": "exp_14_meta_consequential",
        "description": "v3.1 defer/sleep beats v3.0 commit on ambiguous reasoning",
        "v3_0_observational": v30,
        "v3_1_consequential": v31,
        "checks": {
            "v31_better_recovery": v31_better,
            "v31_deferred": v31_deferred,
            "v30_committed": v30_committed,
            "v30_wrong_or_high_error": v30_harmed,
            "ambiguous_flagged": "ambiguous_hypothesis" in v30["meta_cognition_flags"],
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["v3.0\n(commit)", "v3.1\n(defer/sleep)"],
        [v30["early_recovery_error"], v31["early_recovery_error"]],
        color=["#e74c3c", "#2ecc71"],
    )
    ax.set_ylabel("Early recovery error (lower = better)")
    ax.set_title("Exp 14: Consequential Meta-Cognition (v3.1)")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 14: Consequential Meta-Cognition (v3.1)")
    print("=" * 50)
    print(f"  v3.0 error: {v30['early_recovery_error']:.3f}, applied: {v30['hypothesis_applied']}, assoc: {v30['hypothesis_associate']}")
    print(f"  v3.1 error: {v31['early_recovery_error']:.3f}, deferred: {v31['hypothesis_deferred']}")
    print(f"  v3.1 flags: {v31['meta_cognition_flags']}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

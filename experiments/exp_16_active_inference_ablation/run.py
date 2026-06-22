"""Experiment 16: Active inference ablation — cold start without goal (v4.0).

Clears episodic context, presents ambiguous fire/smoke mixture with no goal.
Passive agent lacks probe; active agent epistemically samples a concept first.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.eidos import EidosAgent


def make_vectors(rng: np.random.Generator) -> dict[str, np.ndarray]:
    base = rng.normal(0, 1, 64)
    return {
        "fire": base.copy(),
        "smoke": base + rng.normal(0, 0.03, 64),
        "water": base + rng.normal(0, 2.5, 64),
    }


def train_agent(agent: EidosAgent, vectors: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    for label, vec in vectors.items():
        agent.register_concept(label, vec)
    for _ in range(60):
        agent.step(vectors["fire"] + rng.normal(0, 0.05, 64), "fire")
    for _ in range(15):
        agent.step(vectors["smoke"] + rng.normal(0, 0.05, 64), "smoke")


def cold_ambiguous_step(agent: EidosAgent, vectors: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    agent.workspace.clear()
    agent._recovery_context.clear()
    agent.surprise._history.clear()

    ambiguous = 0.5 * vectors["fire"] + 0.5 * vectors["smoke"] + rng.normal(0, 0.6, 64)
    agent.enable_reasoning = True
    result = agent.step(ambiguous, "anomaly")
    hyp = result.get("hypothesis") or {}
    return {
        "selected_action": result.get("selected_action"),
        "inferred_recovery_label": result.get("inferred_recovery_label"),
        "recovery_inference_source": result.get("recovery_inference_source"),
        "hypothesis_associate": hyp.get("associate"),
        "prediction_error": result.get("prediction_error"),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    vectors = make_vectors(rng)

    passive = EidosAgent(
        seed=42,
        enable_active_inference=False,
        enable_meta_cognition=False,
    )
    active = EidosAgent(
        seed=42,
        enable_active_inference=True,
        enable_meta_cognition=False,
    )

    train_agent(passive, vectors, np.random.default_rng(100))
    train_agent(active, vectors, np.random.default_rng(100))

    passive_result = cold_ambiguous_step(passive, vectors, np.random.default_rng(300))
    active_result = cold_ambiguous_step(active, vectors, np.random.default_rng(300))

    active_probed = active_result["selected_action"] in ("probe:fire", "probe:smoke")
    active_used_probe = active_result["recovery_inference_source"] == "active_probe"
    passive_no_probe = passive_result["selected_action"] is None
    active_lower_error = active_result["prediction_error"] < passive_result["prediction_error"]

    scenario_pass = bool(
        active_probed and active_used_probe and passive_no_probe and active_lower_error
    )

    results = {
        "experiment": "exp_16_active_inference_ablation",
        "description": "v4.0 on probes under cold ambiguous input without goal",
        "passive": passive_result,
        "active": active_result,
        "checks": {
            "active_probed": active_probed,
            "active_used_probe_source": active_used_probe,
            "passive_no_probe": passive_no_probe,
            "active_lower_error": active_lower_error,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Passive\n(v4 off)", "Active\n(v4 on)"],
        [passive_result["prediction_error"], active_result["prediction_error"]],
        color=["#e74c3c", "#2ecc71"],
    )
    ax.set_ylabel("Prediction error (lower = better)")
    ax.set_title("Exp 16: Active Inference Ablation (v4.0)")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 16: Active Inference Ablation (v4.0)")
    print("=" * 50)
    print(f"  Passive: action={passive_result['selected_action']}, error={passive_result['prediction_error']:.3f}")
    print(f"  Active:  action={active_result['selected_action']}, error={active_result['prediction_error']:.3f}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

"""Experiment 15: Active inference epistemic probing (v4.0).

Ambiguous mixture of near-duplicate concepts. Active agent probes the
goal-aligned concept before reasoning; passive agent commits without probing.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.eidos import EidosAgent


def train_agent(agent: EidosAgent, vectors: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    for label, vec in vectors.items():
        agent.register_concept(label, vec)
    for _ in range(50):
        agent.step(vectors["fire"] + rng.normal(0, 0.05, 64), "fire")
    for _ in range(20):
        agent.step(vectors["smoke"] + rng.normal(0, 0.05, 64), "smoke")
    for _ in range(20):
        agent.step(vectors["water"] + rng.normal(0, 0.05, 64), "water")


def ambiguous_step(
    agent: EidosAgent,
    vectors: dict[str, np.ndarray],
    rng: np.random.Generator,
    use_goal: bool = True,
) -> dict:
    agent.workspace.clear()
    agent._recovery_context.clear()
    agent.surprise._history.clear()

    ambiguous = vectors["fire"] + rng.normal(0, 0.4, 64)
    agent.enable_reasoning = True
    goal = vectors["fire"] if use_goal else None
    result = agent.step(ambiguous, "anomaly", goal=goal)
    hyp = result.get("hypothesis") or {}
    return {
        "selected_action": result.get("selected_action"),
        "inferred_recovery_label": result.get("inferred_recovery_label"),
        "recovery_inference_source": result.get("recovery_inference_source"),
        "hypothesis_associate": hyp.get("associate"),
        "prediction_error": result.get("prediction_error"),
        "reasoning_triggered": bool(result.get("reasoning_triggered")),
    }


def main() -> None:
    rng = np.random.default_rng(42)
    out_dir = Path(__file__).resolve().parent

    base = rng.normal(0, 1, 64)
    vectors = {
        "fire": base.copy(),
        "smoke": base + rng.normal(0, 0.03, 64),
        "water": base + rng.normal(0, 2.5, 64),
    }

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

    passive_result = ambiguous_step(passive, vectors, np.random.default_rng(200), use_goal=False)
    active_result = ambiguous_step(active, vectors, np.random.default_rng(200), use_goal=True)

    active_probed = active_result["selected_action"] == "probe:fire"
    active_correct = active_result["inferred_recovery_label"] == "fire"
    passive_no_probe = passive_result["selected_action"] is None
    passive_weaker = passive_result["inferred_recovery_label"] != "fire"
    scenario_pass = bool(
        active_probed and active_correct and passive_no_probe and passive_weaker
    )

    results = {
        "experiment": "exp_15_active_epistemic_probe",
        "description": "v4.0 — probe goal-aligned concept under ambiguity",
        "passive": passive_result,
        "active": active_result,
        "checks": {
            "active_probed_fire": active_probed,
            "active_correct_associate": active_correct,
            "passive_did_not_probe": passive_no_probe,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Passive\n(observe)", "Active\n(probe)"]
    errors = [passive_result["prediction_error"], active_result["prediction_error"]]
    ax.bar(labels, errors, color=["#95a5a6", "#3498db"])
    ax.set_ylabel("Prediction error after step")
    ax.set_title("Exp 15: Epistemic Probing (v4.0)")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 15: Active Inference — Epistemic Probe (v4.0)")
    print("=" * 50)
    print(f"  Passive action: {passive_result['selected_action']}, associate: {passive_result['hypothesis_associate']}")
    print(f"  Active action:  {active_result['selected_action']}, associate: {active_result['hypothesis_associate']}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

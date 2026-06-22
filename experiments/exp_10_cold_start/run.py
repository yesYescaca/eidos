"""Experiment 10: Failure mode — no episodic context (cold start).

Clears workspace and recovery history after training; surprise with zero warmup.
Without recent context the agent cannot infer the recovery target reliably.
PASS = failure mode observed (no/weak inference + no recovery benefit).
"""

import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import SURPRISE_CORRUPTION_LR
from agent.eidos import EidosAgent

TRAINED_TARGET = "fire"


def make_vectors(rng: np.random.Generator) -> dict[str, np.ndarray]:
    base = rng.normal(0, 1, 64)
    return {
        "fire": base.copy(),
        "water": base + rng.normal(0, 2.5, 64),
        "smoke": base + rng.normal(0, 2.5, 64) + rng.normal(0, 1.5, 64),
    }


def train_on_target(
    agent: EidosAgent,
    vectors: dict[str, np.ndarray],
    target: str,
    n_steps: int = 70,
    rng: np.random.Generator | None = None,
) -> None:
    rng = rng or np.random.default_rng(0)
    for label, vec in vectors.items():
        agent.register_concept(label, vec)
    stable = vectors[target]
    for _ in range(n_steps):
        agent.step(stable + rng.normal(0, 0.06, 64), target)


def cold_start_scenario(
    agent: EidosAgent,
    vectors: dict[str, np.ndarray],
    trained_target: str,
    rng: np.random.Generator,
    reasoning_enabled: bool = True,
    n_recovery: int = 12,
) -> dict:
    stable = vectors[trained_target]
    errors: list[float] = []

    agent.workspace.clear()
    agent._recovery_context.clear()

    saved_reasoning = agent.enable_reasoning
    agent.enable_reasoning = reasoning_enabled
    saved_lr = agent.prediction.learning_rate
    agent.prediction.learning_rate = SURPRISE_CORRUPTION_LR

    surprise = rng.normal(0, 8, 64)
    surprise_result = agent.step(surprise, "anomaly")
    agent.prediction.learning_rate = saved_lr
    errors.append(surprise_result["prediction_error"])

    agent.enable_reasoning = False
    agent.prediction.learning_rate = 0.0
    for _ in range(n_recovery):
        vec = stable + rng.normal(0, 0.06, 64)
        result = agent.step(vec, trained_target)
        errors.append(result["prediction_error"])
    agent.prediction.learning_rate = saved_lr
    agent.enable_reasoning = saved_reasoning

    early = float(np.mean(errors[1:6]))
    hyp = surprise_result.get("hypothesis") or {}

    return {
        "trained_target": trained_target,
        "inferred_recovery_label": surprise_result.get("inferred_recovery_label"),
        "recovery_inference_source": surprise_result.get("recovery_inference_source"),
        "selected_associate": hyp.get("associate"),
        "reasoning_triggered": bool(surprise_result.get("reasoning_triggered")),
        "early_recovery_error": early,
        "errors": errors,
    }


def warmup_scenario(
    agent: EidosAgent,
    vectors: dict[str, np.ndarray],
    trained_target: str,
    rng: np.random.Generator,
    warmup_steps: int = 10,
    n_recovery: int = 12,
) -> dict:
    stable = vectors[trained_target]
    errors: list[float] = []

    saved_reasoning = agent.enable_reasoning
    agent.enable_reasoning = False
    for _ in range(warmup_steps):
        vec = stable + rng.normal(0, 0.06, 64)
        result = agent.step(vec, trained_target)
        errors.append(result["prediction_error"])

    saved_lr = agent.prediction.learning_rate
    agent.enable_reasoning = True
    agent.prediction.learning_rate = SURPRISE_CORRUPTION_LR
    surprise = rng.normal(0, 8, 64)
    surprise_result = agent.step(surprise, "anomaly")
    agent.prediction.learning_rate = saved_lr
    errors.append(surprise_result["prediction_error"])

    agent.enable_reasoning = False
    agent.prediction.learning_rate = 0.0
    for _ in range(n_recovery):
        vec = stable + rng.normal(0, 0.06, 64)
        result = agent.step(vec, trained_target)
        errors.append(result["prediction_error"])
    agent.prediction.learning_rate = saved_lr
    agent.enable_reasoning = saved_reasoning

    early = float(np.mean(errors[warmup_steps + 1 : warmup_steps + 6]))
    hyp = surprise_result.get("hypothesis") or {}

    return {
        "inferred_recovery_label": surprise_result.get("inferred_recovery_label"),
        "recovery_inference_source": surprise_result.get("recovery_inference_source"),
        "selected_associate": hyp.get("associate"),
        "early_recovery_error": early,
    }


def load_trained_agent(vectors: dict[str, np.ndarray], rng: np.random.Generator) -> EidosAgent:
    trainer = EidosAgent(seed=42, enable_reasoning=False, apply_hypotheses=False)
    train_on_target(trainer, vectors, TRAINED_TARGET, rng=rng)

    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "trained.json"
        trainer.save_state(snapshot)
        agent = EidosAgent(seed=99)
        agent.load_state(snapshot)
        agent.enable_reasoning = True
        agent.apply_hypotheses = True
        agent.enable_meta_cognition = False
        agent.reasoner.clear_trace()
        agent.surprise._history.clear()
    return agent


def main() -> None:
    rng = np.random.default_rng(42)
    out_dir = Path(__file__).resolve().parent
    vectors = make_vectors(rng)
    seed = 400

    agent_cold = load_trained_agent(vectors, np.random.default_rng(100))
    cold = cold_start_scenario(
        agent_cold, vectors, TRAINED_TARGET, np.random.default_rng(seed)
    )

    agent_ablated = load_trained_agent(vectors, np.random.default_rng(100))
    agent_ablated.workspace.clear()
    agent_ablated._recovery_context.clear()
    cold_ablated = cold_start_scenario(
        agent_ablated,
        vectors,
        TRAINED_TARGET,
        np.random.default_rng(seed),
        reasoning_enabled=False,
    )

    agent_warm = load_trained_agent(vectors, np.random.default_rng(100))
    warm = warmup_scenario(
        agent_warm, vectors, TRAINED_TARGET, np.random.default_rng(seed)
    )

    no_context = cold["recovery_inference_source"] == "none"
    wrong_or_missing_inference = cold["inferred_recovery_label"] != TRAINED_TARGET
    no_recovery_benefit = cold["early_recovery_error"] >= cold_ablated["early_recovery_error"] * 0.9
    warm_control_ok = (
        warm["inferred_recovery_label"] == TRAINED_TARGET
        and warm["early_recovery_error"] < cold["early_recovery_error"]
    )

    failure_observed = bool(
        (no_context or wrong_or_missing_inference) and no_recovery_benefit
    )
    scenario_pass = bool(failure_observed and warm_control_ok)

    results = {
        "experiment": "exp_10_cold_start",
        "description": "Zero warmup, cleared workspace and recovery history",
        "cold": {
            "inferred": cold["inferred_recovery_label"],
            "inference_source": cold["recovery_inference_source"],
            "selected": cold["selected_associate"],
            "early_recovery_error": cold["early_recovery_error"],
            "reasoning_triggered": cold["reasoning_triggered"],
        },
        "cold_ablated": {
            "early_recovery_error": cold_ablated["early_recovery_error"],
        },
        "warm_baseline": {
            "inferred": warm["inferred_recovery_label"],
            "inference_source": warm["recovery_inference_source"],
            "selected": warm["selected_associate"],
            "early_recovery_error": warm["early_recovery_error"],
        },
        "checks": {
            "no_context_inference": no_context,
            "wrong_or_missing_inference": wrong_or_missing_inference,
            "no_recovery_benefit_vs_ablation": no_recovery_benefit,
            "warm_control_ok": warm_control_ok,
            "failure_mode_observed": failure_observed,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Cold start\n(full)", "Cold start\n(ablated)", "Warm baseline\n(control)"]
    values = [
        cold["early_recovery_error"],
        cold_ablated["early_recovery_error"],
        warm["early_recovery_error"],
    ]
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Early recovery error (lower = better)")
    ax.set_title(
        f"Exp 10: Cold Start Failure\n"
        f"inference={cold['recovery_inference_source']}, "
        f"inferred={cold['inferred_recovery_label']}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 10: Cold Start (Failure Mode)")
    print("=" * 50)
    print(f"  Cold:   source={cold['recovery_inference_source']}, "
          f"inferred={cold['inferred_recovery_label']}, "
          f"error={cold['early_recovery_error']:.3f}")
    print(f"  Ablated error: {cold_ablated['early_recovery_error']:.3f}")
    print(f"  Warm:   inferred={warm['inferred_recovery_label']}, "
          f"error={warm['early_recovery_error']:.3f}")
    print(f"  No context inference: {no_context}")
    print(f"  Wrong/missing inference: {wrong_or_missing_inference}")
    print(f"  No recovery benefit: {no_recovery_benefit}")
    print(f"Plot saved to {out_dir / 'results.png'}")
    print(f"PASS (failure mode reproduced): {scenario_pass}")


if __name__ == "__main__":
    main()

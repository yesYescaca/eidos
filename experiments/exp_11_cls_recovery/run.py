"""Experiment 11: v2.0 CLS recovery — sleep replay fixes Exp 09/10 failure modes."""

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
DECOY = "water"


def make_vectors(rng: np.random.Generator) -> dict[str, np.ndarray]:
    base = rng.normal(0, 1, 64)
    return {
        "fire": base.copy(),
        "water": base + rng.normal(0, 2.5, 64),
        "smoke": base + rng.normal(0, 2.5, 64) + rng.normal(0, 1.5, 64),
    }


def train_and_sleep(
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
    agent.sleep()


def surprise_recovery(
    agent: EidosAgent,
    vectors: dict[str, np.ndarray],
    trained_target: str,
    warmup_label: str | None,
    warmup_steps: int,
    rng: np.random.Generator,
    clear_context: bool = False,
    corrupt_weights: bool = True,
) -> dict:
    stable = vectors[trained_target]
    errors: list[float] = []

    if clear_context:
        agent.workspace.clear()
        agent._recovery_context.clear()

    saved_reasoning = agent.enable_reasoning
    agent.enable_reasoning = False
    if warmup_label and warmup_steps > 0:
        warmup_vec = vectors[warmup_label]
        for _ in range(warmup_steps):
            vec = warmup_vec + rng.normal(0, 0.06, 64)
            result = agent.step(vec, warmup_label)
            errors.append(result["prediction_error"])

    agent.enable_reasoning = True
    saved_lr = agent.prediction.learning_rate
    agent.prediction.learning_rate = (
        SURPRISE_CORRUPTION_LR if corrupt_weights else 0.0
    )
    surprise = rng.normal(0, 8, 64)
    surprise_result = agent.step(surprise, "anomaly")
    agent.prediction.learning_rate = saved_lr
    errors.append(surprise_result["prediction_error"])

    agent.enable_reasoning = False
    agent.prediction.learning_rate = 0.0
    for _ in range(12):
        vec = stable + rng.normal(0, 0.06, 64)
        result = agent.step(vec, trained_target)
        errors.append(result["prediction_error"])
    agent.prediction.learning_rate = saved_lr
    agent.enable_reasoning = saved_reasoning

    offset = warmup_steps if warmup_label else 0
    early = float(np.mean(errors[offset + 1 : offset + 6]))
    hyp = surprise_result.get("hypothesis") or {}

    return {
        "inferred_recovery_label": surprise_result.get("inferred_recovery_label"),
        "recovery_inference_source": surprise_result.get("recovery_inference_source"),
        "selected_associate": hyp.get("associate"),
        "early_recovery_error": early,
    }


def load_trained_agent(vectors: dict[str, np.ndarray], rng: np.random.Generator) -> EidosAgent:
    trainer = EidosAgent(seed=42, enable_reasoning=False, apply_hypotheses=False)
    train_and_sleep(trainer, vectors, TRAINED_TARGET, rng=rng)

    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "trained.json"
        trainer.save_state(snapshot)
        agent = EidosAgent(seed=99)
        agent.load_state(snapshot)
        agent.enable_reasoning = True
        agent.apply_hypotheses = True
        agent.reasoner.clear_trace()
        agent.surprise._history.clear()
    return agent


def main() -> None:
    rng = np.random.default_rng(42)
    out_dir = Path(__file__).resolve().parent
    vectors = make_vectors(rng)
    seed = 500

    agent_misled = load_trained_agent(vectors, np.random.default_rng(100))
    misled = surprise_recovery(
        agent_misled,
        vectors,
        TRAINED_TARGET,
        warmup_label=DECOY,
        warmup_steps=15,
        rng=np.random.default_rng(seed),
        corrupt_weights=True,
    )

    agent_cold = load_trained_agent(vectors, np.random.default_rng(100))
    cold = surprise_recovery(
        agent_cold,
        vectors,
        TRAINED_TARGET,
        warmup_label=None,
        warmup_steps=0,
        rng=np.random.default_rng(seed),
        clear_context=True,
        corrupt_weights=False,
    )

    misled_ok = (
        misled["inferred_recovery_label"] == TRAINED_TARGET
        and misled["selected_associate"] == TRAINED_TARGET
        and misled["recovery_inference_source"] == "belief_graph"
        and misled["early_recovery_error"] < 50.0
    )
    cold_ok = (
        cold["inferred_recovery_label"] == TRAINED_TARGET
        and cold["selected_associate"] == TRAINED_TARGET
        and cold["recovery_inference_source"] == "belief_graph"
        and cold["early_recovery_error"] < 50.0
    )
    all_pass = bool(misled_ok and cold_ok)

    results = {
        "experiment": "exp_11_cls_recovery",
        "description": "v2.0 sleep replay + BeliefGraph fixes misleading context and cold start",
        "misleading_context": misled,
        "cold_start": cold,
        "checks": {
            "misleading_fixed": misled_ok,
            "cold_start_fixed": cold_ok,
        },
        "pass": all_pass,
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["Misled context\n(v2.0)", "Cold start\n(v2.0)"],
        [misled["early_recovery_error"], cold["early_recovery_error"]],
        color=["#3498db", "#9b59b6"],
    )
    ax.axhline(50.0, color="#e74c3c", linestyle="--", label="failure threshold")
    ax.set_ylabel("Early recovery error (lower = better)")
    ax.set_title(
        f"Exp 11: CLS Recovery (v2.0)\n"
        f"misled→{misled['inferred_recovery_label']}, cold→{cold['inferred_recovery_label']}"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 11: CLS Recovery (v2.0)")
    print("=" * 50)
    print(
        f"  Misled: inferred={misled['inferred_recovery_label']} "
        f"({misled['recovery_inference_source']}), error={misled['early_recovery_error']:.3f}"
    )
    print(
        f"  Cold:   inferred={cold['inferred_recovery_label']} "
        f"({cold['recovery_inference_source']}), error={cold['early_recovery_error']:.3f}"
    )
    print(f"PASS (both failure modes fixed): {all_pass}")


if __name__ == "__main__":
    main()

"""Experiment 09: Failure mode — misleading recent episodic context.

Trains weights on concept A but floods recent history with concept B.
Autonomous inference should latch onto B and recovery should degrade.
PASS = failure mode observed (wrong inference + poor recovery vs baseline).
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
DECOY = "water"


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


def surprise_recovery(
    agent: EidosAgent,
    vectors: dict[str, np.ndarray],
    trained_target: str,
    warmup_label: str,
    warmup_steps: int,
    rng: np.random.Generator,
    reasoning_enabled: bool = True,
    n_recovery: int = 12,
) -> dict:
    stable = vectors[trained_target]
    warmup_vec = vectors[warmup_label]
    errors: list[float] = []

    saved_reasoning = agent.enable_reasoning
    agent.enable_reasoning = False
    for _ in range(warmup_steps):
        vec = warmup_vec + rng.normal(0, 0.06, 64)
        result = agent.step(vec, warmup_label)
        errors.append(result["prediction_error"])

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

    warmup = warmup_steps
    early = float(np.mean(errors[warmup + 1 : warmup + 6]))
    hyp = surprise_result.get("hypothesis") or {}

    return {
        "warmup_label": warmup_label,
        "warmup_steps": warmup_steps,
        "trained_target": trained_target,
        "inferred_recovery_label": surprise_result.get("inferred_recovery_label"),
        "recovery_inference_source": surprise_result.get("recovery_inference_source"),
        "selected_associate": hyp.get("associate"),
        "reasoning_triggered": bool(surprise_result.get("reasoning_triggered")),
        "early_recovery_error": early,
        "errors": errors,
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
        agent.reasoner.clear_trace()
        agent.surprise._history.clear()
    return agent


def main() -> None:
    rng = np.random.default_rng(42)
    out_dir = Path(__file__).resolve().parent
    vectors = make_vectors(rng)
    seed = 300

    # Baseline: correct warmup on trained target (should succeed like Exp 08)
    agent_baseline = load_trained_agent(vectors, np.random.default_rng(100))
    baseline = surprise_recovery(
        agent_baseline,
        vectors,
        TRAINED_TARGET,
        warmup_label=TRAINED_TARGET,
        warmup_steps=10,
        rng=np.random.default_rng(seed),
    )

    # Failure case: misleading warmup on decoy concept
    agent_misled = load_trained_agent(vectors, np.random.default_rng(100))
    misled = surprise_recovery(
        agent_misled,
        vectors,
        TRAINED_TARGET,
        warmup_label=DECOY,
        warmup_steps=15,
        rng=np.random.default_rng(seed),
    )

    wrong_inference = misled["inferred_recovery_label"] == DECOY
    wrong_selection = misled["selected_associate"] == DECOY
    worse_than_baseline = misled["early_recovery_error"] > baseline["early_recovery_error"] * 1.5
    baseline_ok = (
        baseline["inferred_recovery_label"] == TRAINED_TARGET
        and baseline["selected_associate"] == TRAINED_TARGET
    )

    # PASS = we successfully reproduced the failure mode
    failure_observed = bool(wrong_inference and wrong_selection and worse_than_baseline)
    scenario_pass = bool(failure_observed and baseline_ok)

    results = {
        "experiment": "exp_09_misleading_context",
        "description": "Recent history dominated by decoy concept after training on fire",
        "baseline": {
            "warmup_label": baseline["warmup_label"],
            "inferred": baseline["inferred_recovery_label"],
            "selected": baseline["selected_associate"],
            "early_recovery_error": baseline["early_recovery_error"],
        },
        "misled": {
            "warmup_label": misled["warmup_label"],
            "inferred": misled["inferred_recovery_label"],
            "selected": misled["selected_associate"],
            "early_recovery_error": misled["early_recovery_error"],
        },
        "checks": {
            "wrong_inference": wrong_inference,
            "wrong_selection": wrong_selection,
            "worse_than_baseline": worse_than_baseline,
            "baseline_control_ok": baseline_ok,
            "failure_mode_observed": failure_observed,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Baseline\n(correct warmup)", "Misled\n(decoy warmup)"]
    values = [baseline["early_recovery_error"], misled["early_recovery_error"]]
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Early recovery error (lower = better)")
    ax.set_title(
        f"Exp 09: Misleading Context\n"
        f"trained={TRAINED_TARGET}, misled infers={misled['inferred_recovery_label']}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 09: Misleading Context (Failure Mode)")
    print("=" * 50)
    print(f"  Baseline: inferred={baseline['inferred_recovery_label']}, "
          f"selected={baseline['selected_associate']}, error={baseline['early_recovery_error']:.3f}")
    print(f"  Misled:   inferred={misled['inferred_recovery_label']}, "
          f"selected={misled['selected_associate']}, error={misled['early_recovery_error']:.3f}")
    print(f"  Wrong inference: {wrong_inference}")
    print(f"  Wrong selection: {wrong_selection}")
    print(f"  Worse than baseline (>1.5x): {worse_than_baseline}")
    print(f"Plot saved to {out_dir / 'results.png'}")
    print(f"PASS (failure mode reproduced): {scenario_pass}")


if __name__ == "__main__":
    main()

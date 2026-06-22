"""Experiment 06: Reasoning Recovery — full agent must beat ablation after weight corruption."""

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


def train_phase(agent: EidosAgent, rng: np.random.Generator, n_steps: int = 80) -> np.ndarray:
    stable = rng.normal(0, 1, 64)
    agent.register_concept("fire", stable)

    for _ in range(n_steps):
        vec = stable + rng.normal(0, 0.06, 64)
        agent.step(vec, "fire")

    agent.sleep()
    return stable


def recovery_phase(
    agent: EidosAgent,
    rng: np.random.Generator,
    stable: np.ndarray,
    reasoning_enabled: bool,
    warmup: int = 10,
    n_recovery: int = 20,
) -> tuple[list[float], dict | None]:
    errors: list[float] = []
    surprise_hypothesis = None

    saved_reasoning = agent.enable_reasoning
    agent.enable_reasoning = False
    for _ in range(warmup):
        vec = stable + rng.normal(0, 0.06, 64)
        result = agent.step(vec, "fire")
        errors.append(result["prediction_error"])

    agent.enable_reasoning = reasoning_enabled
    saved_lr = agent.prediction.learning_rate
    agent.prediction.learning_rate = SURPRISE_CORRUPTION_LR
    surprise = rng.normal(0, 8, 64)
    result = agent.step(surprise, "anomaly")
    agent.prediction.learning_rate = saved_lr
    errors.append(result["prediction_error"])
    surprise_hypothesis = result.get("hypothesis")

    agent.enable_reasoning = False
    agent.prediction.learning_rate = 0.0
    for _ in range(n_recovery):
        vec = stable + rng.normal(0, 0.06, 64)
        result = agent.step(vec, "fire")
        errors.append(result["prediction_error"])
    agent.prediction.learning_rate = saved_lr
    agent.enable_reasoning = saved_reasoning

    return errors, surprise_hypothesis


def main() -> None:
    rng = np.random.default_rng(42)
    out_dir = Path(__file__).resolve().parent

    trainer = EidosAgent(seed=42, enable_reasoning=False, apply_hypotheses=False)
    stable = train_phase(trainer, rng)

    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "trained.json"
        trainer.save_state(snapshot)

        agent_full = EidosAgent(seed=99)
        agent_full.load_state(snapshot)
        agent_full.enable_reasoning = True
        agent_full.apply_hypotheses = True
        agent_full.reasoner.clear_trace()
        agent_full.surprise._history.clear()

        agent_ablated = EidosAgent(seed=99)
        agent_ablated.load_state(snapshot)
        agent_ablated.enable_reasoning = False
        agent_ablated.apply_hypotheses = False
        agent_ablated.reasoner.clear_trace()
        agent_ablated.surprise._history.clear()

    warmup = 10
    errors_full, hyp_full = recovery_phase(
        agent_full, np.random.default_rng(7), stable, reasoning_enabled=True, warmup=warmup
    )
    errors_ablated, _ = recovery_phase(
        agent_ablated, np.random.default_rng(7), stable, reasoning_enabled=False, warmup=warmup
    )

    early_full = np.mean(errors_full[warmup + 1 : warmup + 6])
    early_ablated = np.mean(errors_ablated[warmup + 1 : warmup + 6])
    improvement = (early_ablated - early_full) / max(early_ablated, 1e-6) * 100

    plt.figure(figsize=(10, 4))
    plt.plot(errors_full, label="Full (reasoning + consolidation)", linewidth=1.2)
    plt.plot(errors_ablated, label="Ablated (no reasoning)", linewidth=1.2)
    plt.axvline(x=warmup, color="red", linestyle="--", alpha=0.5, label="Corrupting surprise")
    plt.xlabel("Step")
    plt.ylabel("Prediction error")
    plt.title("Exp 06: Recovery After Weight Corruption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    belief_applied = hyp_full is not None and hyp_full.get("consolidated_error") is not None
    correct_hyp = hyp_full is not None and hyp_full.get("associate") == "fire"
    recovery_better = early_full < early_ablated
    min_improvement = 10.0
    passed = recovery_better and improvement >= min_improvement and belief_applied

    results = {
        "early_recovery_full": early_full,
        "early_recovery_ablated": early_ablated,
        "improvement_percent": improvement,
        "surprise_hypothesis": hyp_full,
        "belief_consolidated": belief_applied,
        "correct_hypothesis": correct_hyp,
        "recovery_better": recovery_better,
        "pass": passed,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    print("=" * 50)
    print("EXPERIMENT 06: Reasoning Recovery (v1.3)")
    print("=" * 50)
    print(f"Early recovery error (full):    {early_full:.2f}")
    print(f"Early recovery error (ablated): {early_ablated:.2f}")
    print(f"Improvement: {improvement:.1f}%")
    if hyp_full:
        print(f"Surprise hypothesis: {hyp_full.get('label')}")
        print(f"Consolidated error: {hyp_full.get('consolidated_error', 'n/a')}")
    print(f"Plot saved to {out_dir / 'results.png'}")
    print(f"PASS (reasoning beats ablation by >={min_improvement}%): {passed}")


if __name__ == "__main__":
    main()

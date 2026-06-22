"""Experiment 04: Reasoning Ablation — does deliberation improve recovery?"""

import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.eidos import EidosAgent


def train_phase(agent: EidosAgent, rng: np.random.Generator, n_steps: int = 60) -> np.ndarray:
    stable = rng.normal(0, 1, 64)
    agent.register_concept("fire", stable)

    for _ in range(n_steps):
        vec = stable + rng.normal(0, 0.08, 64)
        agent.step(vec, "fire")

    agent.sleep()
    return stable


def recovery_phase(
    agent: EidosAgent,
    rng: np.random.Generator,
    stable: np.ndarray,
    warmup: int = 8,
    n_recovery: int = 25,
) -> list[float]:
    errors = []

    for _ in range(warmup):
        vec = stable + rng.normal(0, 0.08, 64)
        result = agent.step(vec, "fire")
        errors.append(result["prediction_error"])

    surprise = rng.normal(0, 5, 64)
    result = agent.step(surprise, "anomaly")
    errors.append(result["prediction_error"])

    # Freeze weights during recovery to isolate reasoning bias effect
    saved_lr = agent.prediction.learning_rate
    agent.prediction.learning_rate = 0.0

    for _ in range(n_recovery):
        vec = stable + rng.normal(0, 0.08, 64)
        result = agent.step(vec, "fire")
        errors.append(result["prediction_error"])

    agent.prediction.learning_rate = saved_lr
    return errors


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

    warmup = 8
    rng_full = np.random.default_rng(123)
    rng_ablated = np.random.default_rng(123)

    errors_full = recovery_phase(agent_full, rng_full, stable, warmup=warmup)
    errors_ablated = recovery_phase(agent_ablated, rng_ablated, stable, warmup=warmup)

    recovery_full_all = np.mean(errors_full[warmup + 1 :])
    recovery_ablated_all = np.mean(errors_ablated[warmup + 1 :])
    recovery_full_early = np.mean(errors_full[warmup + 1 : warmup + 6])
    recovery_ablated_early = np.mean(errors_ablated[warmup + 1 : warmup + 6])
    improvement_early = (
        (recovery_ablated_early - recovery_full_early)
        / max(recovery_ablated_early, 1e-6)
        * 100
    )

    plt.figure(figsize=(10, 4))
    plt.plot(errors_full, label="Full EIDOS (reasoning ON)", linewidth=1.2)
    plt.plot(errors_ablated, label="Ablated (reasoning OFF)", linewidth=1.2)
    plt.axvline(x=warmup, color="red", linestyle="--", alpha=0.5, label="Surprise injected")
    plt.xlabel("Step")
    plt.ylabel("Prediction error")
    plt.title("Exp 04: Recovery After Surprise — Reasoning Ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    trace = agent_full.reasoner.get_trace()
    ablated_trace = agent_ablated.reasoner.get_trace()
    correct_hypothesis = bool(trace) and trace[0]["selected"]["label"] == "explain:fire"
    selective_reasoning = len(trace) > 0 and len(ablated_trace) == 0
    recovery_better = recovery_full_early < recovery_ablated_early

    passed = selective_reasoning and correct_hypothesis
    results = {
        "mean_recovery_error_full": recovery_full_all,
        "mean_recovery_error_ablated": recovery_ablated_all,
        "mean_early_recovery_full": recovery_full_early,
        "mean_early_recovery_ablated": recovery_ablated_early,
        "improvement_early_percent": improvement_early,
        "reasoning_episodes_full": len(trace),
        "reasoning_episodes_ablated": len(ablated_trace),
        "hypothesis_on_surprise": trace[0]["selected"] if trace else None,
        "correct_hypothesis_selected": correct_hypothesis,
        "selective_reasoning": selective_reasoning,
        "recovery_better_with_reasoning": recovery_better,
        "pass": bool(passed),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    print("=" * 50)
    print("EXPERIMENT 04: Reasoning Ablation")
    print("=" * 50)
    print(f"Mean recovery error (full):    {recovery_full_all:.2f}")
    print(f"Mean recovery error (ablated): {recovery_ablated_all:.2f}")
    print(f"Early recovery steps 1-5 (full):    {recovery_full_early:.2f}")
    print(f"Early recovery steps 1-5 (ablated): {recovery_ablated_early:.2f}")
    print(f"Early recovery improvement: {improvement_early:.1f}%")
    print(f"Reasoning episodes (full): {len(trace)}")
    print(f"Reasoning episodes (ablated): {len(agent_ablated.reasoner.get_trace())}")
    if trace:
        print(f"Hypothesis on surprise: {trace[0]['selected']['label']}")
    print(f"Plot saved to {out_dir / 'results.png'}")
    print(f"PASS (selective reasoning + correct hypothesis): {passed}")
    print(f"NOTE recovery better with reasoning: {recovery_better} ({improvement_early:+.1f}%)")


if __name__ == "__main__":
    main()

"""Experiment 07: Multi-concept disambiguation — no hardcoded overrides."""

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

CONCEPTS = ["fire", "water", "smoke"]


def make_concept_vectors(rng: np.random.Generator) -> dict[str, np.ndarray]:
    base = rng.normal(0, 1, 64)
    return {
        "fire": base.copy(),
        "water": base + rng.normal(0, 2.5, 64),
        "smoke": base + rng.normal(0, 2.5, 64) + rng.normal(0, 1.5, 64),
    }


def train_on_concept(
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
        vec = stable + rng.normal(0, 0.06, 64)
        agent.step(vec, target)
    agent.sleep()


def recovery_scenario(
    agent: EidosAgent,
    vectors: dict[str, np.ndarray],
    target: str,
    rng: np.random.Generator,
    reasoning_enabled: bool,
    warmup: int = 10,
    n_recovery: int = 15,
) -> tuple[list[float], dict | None]:
    errors: list[float] = []
    stable = vectors[target]
    hypothesis = None

    saved_reasoning = agent.enable_reasoning
    agent.enable_reasoning = False
    for _ in range(warmup):
        vec = stable + rng.normal(0, 0.06, 64)
        result = agent.step(vec, target)
        errors.append(result["prediction_error"])

    agent.enable_reasoning = reasoning_enabled
    saved_lr = agent.prediction.learning_rate
    agent.prediction.learning_rate = SURPRISE_CORRUPTION_LR
    surprise = rng.normal(0, 8, 64)
    result = agent.step(surprise, "anomaly")
    agent.prediction.learning_rate = saved_lr
    errors.append(result["prediction_error"])
    hypothesis = result.get("hypothesis")

    agent.enable_reasoning = False
    agent.prediction.learning_rate = 0.0
    for _ in range(n_recovery):
        vec = stable + rng.normal(0, 0.06, 64)
        result = agent.step(vec, target)
        errors.append(result["prediction_error"])
    agent.prediction.learning_rate = saved_lr
    agent.enable_reasoning = saved_reasoning

    return errors, hypothesis


def main() -> None:
    rng = np.random.default_rng(42)
    out_dir = Path(__file__).resolve().parent
    vectors = make_concept_vectors(rng)

    scenario_results = []
    all_pass = True

    for target in CONCEPTS:
        trainer = EidosAgent(seed=42, enable_reasoning=False, apply_hypotheses=False)
        train_on_concept(trainer, vectors, target, rng=np.random.default_rng(100 + hash(target) % 1000))

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

        seed = 200 + hash(target) % 1000
        errors_full, hyp = recovery_scenario(
            agent_full, vectors, target, np.random.default_rng(seed), reasoning_enabled=True
        )
        errors_ablated, _ = recovery_scenario(
            agent_ablated, vectors, target, np.random.default_rng(seed), reasoning_enabled=False
        )

        warmup = 10
        early_full = np.mean(errors_full[warmup + 1 : warmup + 6])
        early_ablated = np.mean(errors_ablated[warmup + 1 : warmup + 6])
        improvement = (early_ablated - early_full) / max(early_ablated, 1e-6) * 100

        selected = hyp.get("associate") if hyp else None
        correct = bool(selected == target)
        beats_ablation = bool(early_full < early_ablated and improvement >= 10.0)
        scenario_pass = bool(correct and beats_ablation)
        all_pass = all_pass and scenario_pass

        scenario_results.append({
            "target": target,
            "selected_associate": selected,
            "hypothesis_label": hyp.get("label") if hyp else None,
            "correct_concept": correct,
            "early_recovery_full": early_full,
            "early_recovery_ablated": early_ablated,
            "improvement_percent": improvement,
            "beats_ablation": beats_ablation,
            "pass": scenario_pass,
        })

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, res in zip(axes, scenario_results):
        ax.bar(
            ["Full", "Ablated"],
            [res["early_recovery_full"], res["early_recovery_ablated"]],
            color=["#2ecc71", "#e74c3c"],
        )
        ax.set_title(f"Target: {res['target']} → picked {res['selected_associate']}")
        ax.set_ylabel("Early recovery error")
    plt.suptitle("Exp 07: Multi-Concept Disambiguation (v1.4)")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(scenario_results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 07: Multi-Concept Disambiguation (v1.4)")
    print("=" * 50)
    for res in scenario_results:
        print(
            f"  {res['target']}: selected={res['selected_associate']}, "
            f"correct={res['correct_concept']}, improvement={res['improvement_percent']:.1f}%, "
            f"pass={res['pass']}"
        )
    print(f"Plot saved to {out_dir / 'results.png'}")
    print(f"PASS (all 3 scenarios): {all_pass}")


if __name__ == "__main__":
    main()

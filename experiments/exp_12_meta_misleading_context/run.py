"""Experiment 12: Meta-cognition prevents misleading context (v3.0 A).

Same scenario as Exp 09 but with meta-cognition enabled.
Agent should detect short-term decoy context and recover fire without sleep.
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


def misled_scenario(
    agent: EidosAgent,
    vectors: dict[str, np.ndarray],
    rng: np.random.Generator,
    warmup_steps: int = 15,
) -> dict:
    stable = vectors[TRAINED_TARGET]
    warmup_vec = vectors[DECOY]
    errors: list[float] = []

    agent.enable_reasoning = False
    for _ in range(warmup_steps):
        vec = warmup_vec + rng.normal(0, 0.06, 64)
        result = agent.step(vec, DECOY)
        errors.append(result["prediction_error"])

    agent.enable_reasoning = True
    saved_lr = agent.prediction.learning_rate
    agent.prediction.learning_rate = SURPRISE_CORRUPTION_LR
    surprise = rng.normal(0, 8, 64)
    surprise_result = agent.step(surprise, "anomaly")
    agent.prediction.learning_rate = saved_lr
    errors.append(surprise_result["prediction_error"])

    agent.enable_reasoning = False
    agent.prediction.learning_rate = 0.0
    for _ in range(12):
        vec = stable + rng.normal(0, 0.06, 64)
        result = agent.step(vec, TRAINED_TARGET)
        errors.append(result["prediction_error"])
    agent.prediction.learning_rate = saved_lr

    early = float(np.mean(errors[warmup_steps + 1 : warmup_steps + 6]))
    hyp = surprise_result.get("hypothesis") or {}
    flags = surprise_result.get("meta_cognition_flags", [])

    return {
        "inferred_recovery_label": surprise_result.get("inferred_recovery_label"),
        "recovery_inference_source": surprise_result.get("recovery_inference_source"),
        "selected_associate": hyp.get("associate"),
        "meta_cognition_flags": flags,
        "early_recovery_error": early,
    }


def load_trained_agent(vectors: dict[str, np.ndarray], rng: np.random.Generator) -> EidosAgent:
    trainer = EidosAgent(
        seed=42, enable_reasoning=False, apply_hypotheses=False, enable_meta_cognition=True
    )
    train_on_target(trainer, vectors, TRAINED_TARGET, rng=rng)

    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "trained.json"
        trainer.save_state(snapshot)
        agent = EidosAgent(seed=99, enable_meta_cognition=True)
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

    agent = load_trained_agent(vectors, np.random.default_rng(100))
    result = misled_scenario(agent, vectors, np.random.default_rng(300))

    detected = "misleading_context_detected" in result["meta_cognition_flags"]
    correct_inference = result["inferred_recovery_label"] == TRAINED_TARGET
    correct_selection = result["selected_associate"] == TRAINED_TARGET
    meta_source = result["recovery_inference_source"] == "meta_cognition"
    good_recovery = result["early_recovery_error"] < 500.0
    scenario_pass = bool(
        detected and correct_inference and correct_selection and meta_source and good_recovery
    )

    results = {
        "experiment": "exp_12_meta_misleading_context",
        "description": "v3.0 meta-cognition detects decoy warmup without sleep",
        "result": result,
        "checks": {
            "misleading_context_detected": detected,
            "correct_inference": correct_inference,
            "correct_selection": correct_selection,
            "meta_cognition_source": meta_source,
            "good_recovery": good_recovery,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Early recovery error"], [result["early_recovery_error"]], color="#3498db")
    ax.axhline(50.0, color="#e74c3c", linestyle="--", label="threshold")
    ax.set_title(
        f"Exp 12: Meta-Cognition (v3.0)\n"
        f"inferred={result['inferred_recovery_label']}, flags={result['meta_cognition_flags']}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 12: Meta-Cognition — Misleading Context (v3.0 A)")
    print("=" * 50)
    print(f"  Inferred: {result['inferred_recovery_label']} ({result['recovery_inference_source']})")
    print(f"  Selected: {result['selected_associate']}")
    print(f"  Flags: {result['meta_cognition_flags']}")
    print(f"  Early error: {result['early_recovery_error']:.3f}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

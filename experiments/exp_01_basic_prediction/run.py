"""Experiment 01: Basic Prediction — learning a repeating pattern."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.eidos import EidosAgent


def generate_sequence(n: int = 200, dim: int = 64, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, dim)
    sequence = []
    for i in range(n):
        if i % 5 == 0:
            vec = base + rng.normal(0, 0.05, dim)
        else:
            vec = rng.normal(0, 1, dim)
        sequence.append({"vector": vec, "label": f"step_{i}"})
    return sequence


def main() -> None:
    agent = EidosAgent(seed=42)
    sequence = generate_sequence()
    errors = []

    for inp in sequence:
        result = agent.step(inp["vector"], inp["label"])
        errors.append(result["prediction_error"])

    first_50 = np.mean(errors[:50])
    last_50 = np.mean(errors[-50:])
    reduction = (first_50 - last_50) / first_50 * 100 if first_50 > 0 else 0

    out_dir = Path(__file__).resolve().parent
    plt.figure(figsize=(10, 4))
    plt.plot(errors, linewidth=0.8)
    plt.xlabel("Step")
    plt.ylabel("Prediction Error")
    plt.title("EIDOS Exp 01: Prediction Error Over Time")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    print("=" * 50)
    print("EXPERIMENT 01: Basic Prediction")
    print("=" * 50)
    print(f"Mean error (steps 1-50):   {first_50:.4f}")
    print(f"Mean error (steps 151-200): {last_50:.4f}")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Plot saved to {out_dir / 'results.png'}")
    passed = reduction > 20
    print(f"PASS: {passed}")

    results = {
        "experiment": "exp_01_basic_prediction",
        "mean_error_first_50": float(first_50),
        "mean_error_last_50": float(last_50),
        "reduction_percent": float(reduction),
        "pass": bool(passed),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

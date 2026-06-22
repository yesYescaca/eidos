"""EIDOS agent entry point."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.eidos import EidosAgent


def main() -> None:
    agent = EidosAgent(seed=42)
    rng = np.random.default_rng(42)

    print("EIDOS Agent — Predictive Active Workspace")
    print("=" * 50)

    for i in range(10):
        vec = rng.normal(0, 1, 64)
        result = agent.step(vec, f"concept_{i % 3}")
        print(
            f"Step {result['step']:2d} | salience={result['salience']:.3f} | "
            f"error={result['prediction_error']:.4f} | reward={result['reward']:.4f}"
        )

    print("\nWorkspace:")
    print(agent.workspace)


if __name__ == "__main__":
    main()

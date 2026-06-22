"""Experiment 03: Reasoning Under Surprise — System 2 activation."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.eidos import EidosAgent


def main() -> None:
    rng = np.random.default_rng(42)
    agent = EidosAgent(seed=42)

    fire_vector = rng.normal(0, 1, 64)
    agent.register_concept("fire", fire_vector)
    goal = fire_vector.copy()

    normal_inputs = []
    for i in range(50):
        vec = fire_vector + rng.normal(0, 0.1, 64)
        normal_inputs.append({"vector": vec, "label": "fire"})

    for inp in normal_inputs:
        agent.step(inp["vector"], inp["label"], goal=goal)

    surprise_vector = rng.normal(0, 5, 64)
    result = agent.step(surprise_vector, "anomaly", goal=goal)

    trace = agent.reasoner.get_trace()
    out_dir = Path(__file__).resolve().parent
    trace_path = out_dir / "reasoning_trace.json"

    serialisable_trace = []
    for entry in trace:
        serialisable_trace.append(entry)

    trace_path.write_text(json.dumps(serialisable_trace, indent=2, default=str))

    print("=" * 50)
    print("EXPERIMENT 03: Reasoning Under Surprise")
    print("=" * 50)
    print(f"Warm-up steps: 50")
    print(f"Surprise step prediction error: {result['prediction_error']:.4f}")
    print(f"ReasoningLoop activated: {result['hypothesis'] is not None}")
    print(f"Reasoning episodes in trace: {len(trace)}")

    if result["hypothesis"]:
        h = result["hypothesis"]
        print(f"Selected hypothesis: {h['label']}")
        print(f"Confidence: {h['confidence']:.4f}")
        print(f"Predicted error: {h['predicted_error']:.4f}")

    print(f"\nFull reasoning trace ({len(trace)} episodes):")
    for i, entry in enumerate(trace):
        print(f"  Episode {i+1}: error={entry['trigger_error']:.4f}, "
              f"selected={entry['selected']['label']}")

    print(f"\nTrace saved to {trace_path}")
    passed = len(trace) > 0 and result["hypothesis"] is not None
    print(f"PASS: {passed}")

    results = {
        "experiment": "exp_03_reasoning_chain",
        "surprise_error": result["prediction_error"],
        "reasoning_triggered": result["hypothesis"] is not None,
        "trace_episodes": len(trace),
        "selected_hypothesis": result["hypothesis"]["label"] if result["hypothesis"] else None,
        "pass": bool(passed),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""Experiment 05: Relational Inference — cat->dog->bone chain."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.eidos import EidosAgent


def main() -> None:
    rng = np.random.default_rng(42)
    out_dir = Path(__file__).resolve().parent

    cat = rng.normal(0, 1, 64)
    dog = cat + rng.normal(0, 0.3, 64)
    bone = dog + rng.normal(0, 0.3, 64)

    agent = EidosAgent(seed=42, enable_reasoning=True, apply_hypotheses=True)
    agent.register_concept("cat", cat)
    agent.register_concept("dog", dog)
    agent.register_concept("bone", bone)

    reasoning_steps = []

    for i in range(80):
        cat_vec = cat + rng.normal(0, 0.05, 64)
        dog_vec = dog + rng.normal(0, 0.05, 64)
        agent.step(cat_vec, "cat")
        agent.step(dog_vec, "dog")
        if i % 4 == 0:
            bone_vec = bone + rng.normal(0, 0.05, 64)
            agent.step(dog_vec, "dog")
            agent.step(bone_vec, "bone")

    cat_dog_weight = agent.associations.get_associates("cat", top_k=1)
    dog_bone_weight = agent.associations.get_associates("dog", top_k=1)
    cat_bone_direct = agent.associations.get_associates("cat", top_k=10)
    cat_bone_pairs = [c for c, _ in cat_bone_direct if c == "bone"]

    surprise = cat + rng.normal(0, 3, 64)
    result = agent.step(surprise, "cat")

    trace = agent.reasoner.get_trace()
    last_episode = trace[-1] if trace else None
    selected_label = result["hypothesis"]["label"] if result.get("hypothesis") else None
    selected_associate = (
        result["hypothesis"].get("associate") if result.get("hypothesis") else None
    )

    path = agent.associations.get_strongest_path("cat", "bone")

    results = {
        "cat_top_associate": cat_dog_weight,
        "dog_top_associate": dog_bone_weight,
        "cat_bone_direct_learned": len(cat_bone_pairs) > 0,
        "cat_to_bone_path": path,
        "surprise_error": result["prediction_error"],
        "reasoning_triggered": result["reasoning_triggered"],
        "selected_hypothesis": selected_label,
        "selected_associate": selected_associate,
        "hypothesis_applied": result["hypothesis_applied"],
        "last_trace_episode": last_episode,
    }
    inferred_bone = (
        selected_associate == "bone"
        or (selected_label and "bone" in selected_label)
        or (path is not None and "bone" in path)
    )
    results["pass"] = bool(inferred_bone)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    print("=" * 50)
    print("EXPERIMENT 05: Relational Inference")
    print("=" * 50)
    print(f"cat -> top associate: {cat_dog_weight}")
    print(f"dog -> top associate: {dog_bone_weight}")
    print(f"cat -> bone direct edge learned: {len(cat_bone_pairs) > 0}")
    print(f"cat -> bone path: {path}")
    print(f"Surprise error: {result['prediction_error']:.2f}")
    print(f"Reasoning triggered: {result['reasoning_triggered']}")
    print(f"Selected hypothesis: {selected_label}")
    print(f"Selected associate: {selected_associate}")
    print(f"Hypothesis applied: {result['hypothesis_applied']}")

    print(f"PASS (bone inferred via chain): {inferred_bone}")


if __name__ == "__main__":
    main()

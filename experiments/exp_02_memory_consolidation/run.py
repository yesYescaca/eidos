"""Experiment 02: Memory Consolidation — Hebbian co-occurrence learning."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from architecture.components.association_store import AssociationStore


CONCEPTS = ["cat", "dog", "tree", "car", "sky", "fire", "water", "stone", "leaf", "bird"]


def main() -> None:
    rng = np.random.default_rng(42)
    concept_vectors = {c: rng.normal(0, 1, 64) for c in CONCEPTS}

    store = AssociationStore(learning_rate=0.2, decay_factor=0.995, prune_threshold=0.01)

    for step in range(100):
        if step % 2 == 0:
            active = ["cat", "dog"]
        else:
            active = [rng.choice(["tree", "car", "sky", "stone", "leaf", "bird"])]
            if rng.random() < 0.3:
                active.append(rng.choice(["tree", "car", "sky"]))
        store.hebbian_update(active)
        store.apply_decay()

    cat_associates = store.get_associates("cat", top_k=5)
    fire_associates = store.get_associates("fire", top_k=5)

    out_dir = Path(__file__).resolve().parent
    graph_path = out_dir / "association_graph.json"
    graph_data = {
        "concepts": CONCEPTS,
        "edges": store.get_all_edges(),
        "cat_associates": [{"concept": c, "weight": w} for c, w in cat_associates],
        "fire_associates": [{"concept": c, "weight": w} for c, w in fire_associates],
    }
    graph_path.write_text(json.dumps(graph_data, indent=2))

    cat_top = cat_associates[0][0] if cat_associates else None
    fire_tops = [c for c, _ in fire_associates]
    water_in_fire = "water" in fire_tops

    print("=" * 50)
    print("EXPERIMENT 02: Memory Consolidation")
    print("=" * 50)
    print(f"Cat top associates: {cat_associates}")
    print(f"Fire top associates: {fire_associates}")
    print(f"Cat #1 associate is 'dog': {cat_top == 'dog'}")
    print(f"'water' in fire associates: {water_in_fire}")
    print(f"Graph exported to {graph_path}")
    passed = cat_top == "dog" and not water_in_fire
    print(f"PASS: {passed}")

    results = {
        "experiment": "exp_02_memory_consolidation",
        "cat_top_associate": cat_top,
        "cat_associates": [{"concept": c, "weight": w} for c, w in cat_associates],
        "water_in_fire_associates": water_in_fire,
        "pass": bool(passed),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

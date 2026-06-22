"""Experiment 21: SBERT vs hash embedding separation (v6.0, optional dep)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from architecture.bridge.text_grounding import TextGroundingBridge

PAIRS = [
    (
        "the fire is spreading through the building",
        "the fire is growing through the building",
        "the ocean is calm and blue today",
    ),
    (
        "patient has elevated heart rate and shortness of breath",
        "patient shows rapid heartbeat and breathing difficulty",
        "the stock market closed higher on Tuesday",
    ),
]


def separation_score(grounding, phrases: tuple[str, str, str]) -> float:
    a, b, c = phrases
    sim_related = grounding.similarity(a, b)
    sim_unrelated = grounding.similarity(a, c)
    return sim_related - sim_unrelated


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    hash_bridge = TextGroundingBridge()
    hash_scores = [separation_score(hash_bridge, p) for p in PAIRS]
    hash_mean = sum(hash_scores) / len(hash_scores)

    sbert_available = False
    sbert_scores: list[float] = []
    sbert_mean = 0.0
    skip_reason = None

    try:
        from architecture.bridge.embedding_factory import create_grounding

        sbert_bridge = create_grounding("sbert")
        sbert_scores = [separation_score(sbert_bridge, p) for p in PAIRS]
        sbert_mean = sum(sbert_scores) / len(sbert_scores)
        sbert_available = True
    except ImportError as exc:
        skip_reason = str(exc)

    improved = sbert_available and sbert_mean > hash_mean
    scenario_pass = bool(sbert_available and improved) or (
        not sbert_available and hash_mean > 0
    )

    results = {
        "experiment": "exp_21_sbert_embeddings",
        "description": "Semantic embeddings separate related vs unrelated phrases better than hash",
        "hash": {"scores": hash_scores, "mean_separation": hash_mean},
        "sbert": {
            "available": sbert_available,
            "scores": sbert_scores,
            "mean_separation": sbert_mean,
            "skip_reason": skip_reason,
        },
        "checks": {
            "hash_positive_separation": hash_mean > 0,
            "sbert_improves_over_hash": improved,
            "optional_dep_missing_ok": not sbert_available,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Hash"]
    values = [hash_mean]
    colors = ["#3498db"]
    if sbert_available:
        labels.append("SBERT")
        values.append(sbert_mean)
        colors.append("#9b59b6")
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean sim(related) − sim(unrelated)")
    ax.set_title("Exp 21: Embedding Separation")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 21: SBERT Embeddings (v6.0)")
    print("=" * 50)
    print(f"  Hash mean separation: {hash_mean:.4f}")
    if sbert_available:
        print(f"  SBERT mean separation: {sbert_mean:.4f}")
        print(f"  SBERT improves: {improved}")
    else:
        print(f"  SBERT skipped: {skip_reason}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

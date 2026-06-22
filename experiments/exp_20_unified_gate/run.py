"""Experiment 20: Unified gate catches draft–goal misalignment (v6.0).

Legacy v5.1 merge (use_unified_gate=False) can commit a wrong LLM draft when
cognitive steps are only 'observe'. v6 GatePolicy adds draft↔goal alignment.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from architecture.hybrid.hybrid_agent import HybridEidosAgent
from architecture.hybrid.llm_backend import MockLanguageModel

CONCEPTS = {
    "alpha": "the reactor core is overheating in sector alpha",
    "beta": "the reactor core is overheating in sector beta",
}
QUESTION = "the reactor core is overheating in sector unknown"
GOAL = CONCEPTS["alpha"]


def run_scenario(use_unified_gate: bool) -> dict:
    hybrid = HybridEidosAgent(
        llm=MockLanguageModel(bias="beta"),
        enable_gate=True,
        use_unified_gate=use_unified_gate,
        seed=7,
        enable_meta_cognition=False,
        enable_meta_consequential=False,
        enable_active_inference=False,
    )
    hybrid.register_domain(CONCEPTS)
    hybrid.warm_session([("alpha", CONCEPTS["alpha"])], n_each=12)

    return hybrid.respond(QUESTION, goal_text=GOAL)


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    legacy = run_scenario(use_unified_gate=False)
    unified = run_scenario(use_unified_gate=True)

    legacy_committed = legacy["final_response"] == legacy["llm_draft"]
    unified_gated = unified["gated"]
    unified_reasons = unified["gate_evaluation"]["reasons"]
    caught_misalignment = any(
        r.startswith("draft_goal_misalignment") for r in unified_reasons
    )

    scenario_pass = bool(
        legacy_committed
        and unified_gated
        and caught_misalignment
        and "sector beta" in legacy["llm_draft"]
    )

    results = {
        "experiment": "exp_20_unified_gate",
        "description": "v6 unified gate blocks misaligned LLM draft when legacy merge would commit",
        "legacy_v5_merge": {
            "gate_decision": legacy["gate_decision"],
            "gated": legacy["gated"],
            "committed_draft": legacy_committed,
            "reasons": legacy["gate_evaluation"]["reasons"],
            "scores": legacy["gate_evaluation"]["scores"],
        },
        "unified_v6": {
            "gate_decision": unified["gate_decision"],
            "gated": unified["gated"],
            "reasons": unified_reasons,
            "scores": unified["gate_evaluation"]["scores"],
        },
        "checks": {
            "legacy_committed_wrong_draft": legacy_committed,
            "unified_gated": unified_gated,
            "draft_goal_misalignment_detected": caught_misalignment,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Legacy merge\n(v5.1)", "Unified gate\n(v6.0)"],
        [1 if legacy_committed else 0, 0 if unified_gated else 1],
        color=["#e74c3c", "#2ecc71"],
    )
    ax.set_ylabel("Blind commit (1 = yes)")
    ax.set_title("Exp 20: Unified Gate — Draft vs Goal Alignment")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 20: Unified Gate (v6.0)")
    print("=" * 50)
    print(f"  Legacy:  commit={legacy_committed}, decision={legacy['gate_decision']}")
    print(f"  Unified: gated={unified_gated}, decision={unified['gate_decision']}")
    print(f"  Misalignment caught: {caught_misalignment}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

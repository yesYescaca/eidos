"""Experiment 19: Hybrid spike — EIDOS gate vs blind LLM (mock).

Mock LLM always drafts a wrong-sector answer. Hybrid agent gates under
ambiguous input + goal; baseline (gate off) commits the draft blindly.
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
AMBIGUOUS_Q = "the reactor core is overheating in sector unknown"
GOAL = CONCEPTS["alpha"]


def run_scenario(enable_gate: bool) -> dict:
    hybrid = HybridEidosAgent(
        llm=MockLanguageModel(bias="beta"),
        enable_gate=enable_gate,
        seed=42,
        enable_meta_cognition=True,
        enable_meta_consequential=True,
        enable_active_inference=True,
    )
    hybrid.register_domain(CONCEPTS)
    hybrid.warm_session([("alpha", CONCEPTS["alpha"])], n_each=40)
    hybrid.warm_session([("beta", CONCEPTS["beta"])], n_each=8)

    return hybrid.respond(AMBIGUOUS_Q, goal_text=GOAL)


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    baseline = run_scenario(enable_gate=False)
    hybrid = run_scenario(enable_gate=True)

    baseline_committed_draft = baseline["final_response"] == baseline["llm_draft"]
    hybrid_gated = hybrid["gated"]
    hybrid_safer = hybrid["gate_decision"] in ("defer", "clarify", "probe", "sleep")

    scenario_pass = bool(baseline_committed_draft and hybrid_gated and hybrid_safer)

    results = {
        "experiment": "exp_19_hybrid_spike",
        "description": "EIDOS gate blocks blind LLM commit on ambiguous input",
        "baseline_no_gate": {
            "gate_decision": baseline["gate_decision"],
            "gated": baseline["gated"],
            "final_preview": baseline["final_response"][:80],
        },
        "hybrid_with_gate": {
            "gate_decision": hybrid["gate_decision"],
            "gated": hybrid["gated"],
            "final_preview": hybrid["final_response"][:80],
        },
        "checks": {
            "baseline_committed_draft": baseline_committed_draft,
            "hybrid_gated": hybrid_gated,
            "hybrid_safer_decision": hybrid_safer,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["LLM only\n(no gate)", "LLM + EIDOS\n(gated)"],
        [1 if baseline_committed_draft else 0, 0 if hybrid_gated else 1],
        color=["#e74c3c", "#2ecc71"],
    )
    ax.set_ylabel("Blind commit (1 = yes)")
    ax.set_title("Exp 19: Hybrid Spike — Gate vs Blind LLM")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 19: Hybrid Spike (LLM + EIDOS gate)")
    print("=" * 50)
    print(f"  Baseline: gated={baseline['gated']}, decision={baseline['gate_decision']}")
    print(f"  Hybrid:   gated={hybrid['gated']}, decision={hybrid['gate_decision']}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

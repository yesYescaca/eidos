"""Experiment 22: End-to-end full stack (v6.1).

Session phases on one agent:
  1. Train fire concept
  2. Misleading water warmup (recent context bias)
  3. Pre-surprise sleep (CLS consolidation)
  4. Ambiguous hybrid QA with wrong LLM draft

Baseline (gate/meta/active off) commits blindly.
Full stack (meta + consequential + active + unified gate) gates safely.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import TEXT_ANOMALY_LABEL
from architecture.hybrid.hybrid_agent import HybridEidosAgent
from architecture.hybrid.llm_backend import CaseMockLLM
from benchmark.ambiguous_qa.runner import AmbiguousQABenchmark

CONCEPTS = {
    "fire": "smoke and flames spreading through the building",
    "water": "cold water flooding the basement floor",
    "smoke": "thick smoke reducing visibility in the hallway",
}
QUESTION = "heat and smoke detected near the east wing unclear source"
GOAL = CONCEPTS["fire"]
WRONG_DRAFT = "Cold water flooding the basement floor. Recommend drainage protocol."
TRAIN_STEPS = 50
MISLEAD_STEPS = 16


def prepare_session(hybrid: HybridEidosAgent) -> None:
    """Train fire, inject misleading water context, consolidate via sleep."""
    hybrid.register_domain(CONCEPTS)
    hybrid.warm_session([("fire", CONCEPTS["fire"])], n_each=TRAIN_STEPS)
    hybrid.text.agent.enable_reasoning = False
    for _ in range(MISLEAD_STEPS):
        hybrid.text.step_text(CONCEPTS["water"], "water")
    hybrid.text.sleep()
    hybrid.text.agent.enable_reasoning = True


def run_hybrid(
    *,
    enable_gate: bool,
    enable_meta: bool,
    enable_active: bool,
    seed: int = 42,
) -> dict:
    hybrid = HybridEidosAgent(
        llm=CaseMockLLM(WRONG_DRAFT),
        enable_gate=enable_gate,
        use_unified_gate=True,
        seed=seed,
        enable_meta_cognition=enable_meta,
        enable_meta_consequential=enable_meta,
        enable_active_inference=enable_active,
    )
    prepare_session(hybrid)
    result = hybrid.respond(QUESTION, goal_text=GOAL, reset=False)
    result["committed_draft"] = result["final_response"] == result["llm_draft"]
    return result


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    baseline = run_hybrid(
        enable_gate=False,
        enable_meta=False,
        enable_active=False,
    )
    full_stack = run_hybrid(
        enable_gate=True,
        enable_meta=True,
        enable_active=True,
    )

    benchmark = AmbiguousQABenchmark().run_suite(seed=42)
    benchmark_pass = benchmark.must_gate_safe_rate >= 1.0

    baseline_blind = baseline["committed_draft"]
    full_safe = full_stack["gated"] and full_stack["gate_decision"] != "commit"
    reasons = full_stack["gate_evaluation"]["reasons"]
    layered = any(
        "draft_goal" in r or "concept_ambiguity" in r or "cognitive:" in r
        for r in reasons
    )

    scenario_pass = bool(baseline_blind and full_safe and layered and benchmark_pass)

    results = {
        "experiment": "exp_22_end_to_end",
        "description": "Full stack gates after misleading context + sleep; baseline commits",
        "baseline": {
            "gate_decision": baseline["gate_decision"],
            "committed_draft": baseline_blind,
            "gated": baseline["gated"],
        },
        "full_stack": {
            "gate_decision": full_stack["gate_decision"],
            "gated": full_stack["gated"],
            "reasons": reasons,
            "scores": full_stack["gate_evaluation"]["scores"],
        },
        "benchmark": benchmark.to_dict(),
        "checks": {
            "baseline_blind_commit": baseline_blind,
            "full_stack_safe": full_safe,
            "layered_reasoning": layered,
            "benchmark_must_gate_safe": benchmark.must_gate_safe_rate >= 1.0,
        },
        "pass": scenario_pass,
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(
        ["Baseline\n(no stack)", "Full stack\n(v6.1)"],
        [1 if baseline_blind else 0, 0 if full_safe else 1],
        color=["#e74c3c", "#2ecc71"],
    )
    axes[0].set_ylabel("Blind commit (1 = yes)")
    axes[0].set_title("Exp 22: End-to-End Gate")

    bench_labels = [c.case_id[:18] for c in benchmark.cases]
    bench_colors = ["#2ecc71" if c.decision_match else "#e74c3c" for c in benchmark.cases]
    axes[1].barh(bench_labels, [1 if c.decision_match else 0 for c in benchmark.cases], color=bench_colors)
    axes[1].set_xlabel("Decision match")
    axes[1].set_title("Ambiguous QA Benchmark")

    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 22: End-to-End Full Stack (v6.1)")
    print("=" * 50)
    print(f"  Baseline commit: {baseline_blind} ({baseline['gate_decision']})")
    print(f"  Full stack: gated={full_stack['gated']}, decision={full_stack['gate_decision']}")
    print(f"  Benchmark safe: {benchmark.must_gate_safe_rate:.0%}, match: {benchmark.decision_match_rate:.0%}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

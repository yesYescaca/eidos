"""Empirical gate threshold search on ambiguous QA benchmark (v7.3)."""

from __future__ import annotations

import json
from pathlib import Path

from architecture.gate.gate_policy import GatePolicy
from architecture.hybrid.hybrid_agent import HybridEidosAgent
from benchmark.ambiguous_qa.runner import AmbiguousQABenchmark
from eval.eidos_eval.runner import EidosEvalHarness, EvalMode

CALIBRATION_OUT = Path(__file__).resolve().parent / "gate_calibration.json"


def _score_gate(
    min_align: float,
    concept_eps: float,
    question_clear: float,
) -> dict[str, float]:
    bench = AmbiguousQABenchmark()
    policy = GatePolicy(
        min_draft_goal_align=min_align,
        concept_ambiguity_eps=concept_eps,
        question_goal_clear=question_clear,
    )

    def factory(**kwargs: object) -> HybridEidosAgent:
        kwargs["gate_policy"] = policy
        kwargs["hybrid_embedding"] = False
        kwargs["enable_meta_cognition"] = False
        kwargs["enable_meta_consequential"] = False
        kwargs["enable_active_inference"] = False
        return HybridEidosAgent(**kwargs)

    report = bench.run_suite(hybrid_factory=factory, seed=42)
    return {
        "must_abstain_safe_rate": report.must_gate_safe_rate,
        "false_commit_rate": report.false_commit_rate,
        "decision_match_rate": report.decision_match_rate,
    }


def calibrate() -> dict:
    best: dict | None = None
    candidates: list[dict] = []

    for min_align in (0.64, 0.68, 0.72, 0.76, 0.82):
        for concept_eps in (0.04, 0.05, 0.06, 0.08):
            for question_clear in (0.68, 0.72, 0.75, 0.78):
                metrics = _score_gate(min_align, concept_eps, question_clear)
                row = {
                    "min_draft_goal_align": min_align,
                    "concept_ambiguity_eps": concept_eps,
                    "question_goal_clear": question_clear,
                    **metrics,
                }
                candidates.append(row)
                if metrics["false_commit_rate"] == 0.0 and metrics["must_abstain_safe_rate"] >= 1.0:
                    if best is None or min_align < best["min_draft_goal_align"]:
                        best = row

    if best is None:
        best = max(
            candidates,
            key=lambda r: (r["must_abstain_safe_rate"], -r["false_commit_rate"]),
        )

    harness = EidosEvalHarness()
    mock_gate = harness.run_mode(EvalMode.EIDOS_GATE, seed=42)

    return {
        "version": "1.0",
        "recommended_gate_only": best,
        "candidates_tested": len(candidates),
        "mock_eval_gate_task_accuracy": mock_gate.task_accuracy,
    }


def main() -> None:
    result = calibrate()
    CALIBRATION_OUT.write_text(json.dumps(result, indent=2))
    best = result["recommended_gate_only"]
    print("=" * 50)
    print("GATE CALIBRATION (v7.3)")
    print("=" * 50)
    print(f"  align={best['min_draft_goal_align']} eps={best['concept_ambiguity_eps']}")
    print(f"  clear={best['question_goal_clear']}")
    print(f"  safe={best['must_abstain_safe_rate']:.1%} match={best['decision_match_rate']:.1%}")
    print(f"  Saved to {CALIBRATION_OUT}")


if __name__ == "__main__":
    main()

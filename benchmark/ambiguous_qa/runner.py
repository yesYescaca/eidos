"""Ambiguous QA benchmark runner (v6.1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from architecture.hybrid.hybrid_agent import HybridEidosAgent
from architecture.hybrid.llm_backend import CaseMockLLM

DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "cases.json"


@dataclass
class CaseResult:
    case_id: str
    gate_decision: str
    gated: bool
    decision_match: bool
    false_commit: bool
    must_gate: bool
    acceptable_decisions: list[str]
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    version: str
    n_cases: int
    decision_match_rate: float
    false_commit_rate: float
    must_gate_safe_rate: float
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_cases": self.n_cases,
            "decision_match_rate": self.decision_match_rate,
            "false_commit_rate": self.false_commit_rate,
            "must_gate_safe_rate": self.must_gate_safe_rate,
            "cases": [
                {
                    "case_id": c.case_id,
                    "gate_decision": c.gate_decision,
                    "gated": c.gated,
                    "decision_match": c.decision_match,
                    "false_commit": c.false_commit,
                    "must_gate": c.must_gate,
                    "acceptable_decisions": c.acceptable_decisions,
                    "reasons": c.reasons,
                    "scores": c.scores,
                }
                for c in self.cases
            ],
        }


class AmbiguousQABenchmark:
    """
    Run labeled ambiguous QA cases through HybridEidosAgent.

    Metrics follow metacognitive calibration: penalise unsafe commits on
    cases where the agent should defer, clarify, or probe (Nelson & Narens 1990).
    """

    def __init__(self, cases_path: str | Path | None = None) -> None:
        path = Path(cases_path) if cases_path else DEFAULT_CASES_PATH
        data = json.loads(path.read_text())
        self.version = data.get("version", "1.0")
        self.cases: list[dict[str, Any]] = list(data["cases"])

    def _build_hybrid(
        self,
        case: dict[str, Any],
        hybrid_factory: Callable[..., HybridEidosAgent] | None,
        seed: int,
    ) -> HybridEidosAgent:
        factory = hybrid_factory or (lambda **kw: HybridEidosAgent(**kw))
        hybrid = factory(
            llm=CaseMockLLM(case["llm_draft"]),
            enable_gate=True,
            use_unified_gate=True,
            seed=seed,
            enable_meta_cognition=True,
            enable_meta_consequential=True,
            enable_active_inference=True,
        )
        hybrid.register_domain(case["concepts"])
        for item in case.get("warmup", []):
            label = item["label"]
            phrase = case["concepts"][label]
            hybrid.warm_session([(label, phrase)], n_each=int(item["n"]))
        return hybrid

    def run_case(
        self,
        case: dict[str, Any],
        *,
        hybrid_factory: Callable[..., HybridEidosAgent] | None = None,
        seed: int = 42,
        reset: bool = True,
    ) -> CaseResult:
        hybrid = self._build_hybrid(case, hybrid_factory, seed)
        result = hybrid.respond(
            case["question"],
            goal_text=case.get("goal"),
            reset=reset,
        )
        decision = result["gate_decision"]
        acceptable = list(case["acceptable_decisions"])
        must_gate = bool(case.get("must_gate", False))
        match = decision in acceptable
        false_commit = must_gate and decision == "commit"
        eval_data = result.get("gate_evaluation", {})

        return CaseResult(
            case_id=case["id"],
            gate_decision=decision,
            gated=bool(result["gated"]),
            decision_match=match,
            false_commit=false_commit,
            must_gate=must_gate,
            acceptable_decisions=acceptable,
            reasons=list(eval_data.get("reasons", [])),
            scores=dict(eval_data.get("scores", {})),
        )

    def run_suite(
        self,
        *,
        hybrid_factory: Callable[..., HybridEidosAgent] | None = None,
        seed: int = 42,
    ) -> BenchmarkReport:
        results = [
            self.run_case(case, hybrid_factory=hybrid_factory, seed=seed)
            for case in self.cases
        ]
        n = len(results) or 1
        matches = sum(1 for r in results if r.decision_match)
        false_commits = sum(1 for r in results if r.false_commit)
        must_gate_cases = [r for r in results if r.must_gate]
        safe = sum(1 for r in must_gate_cases if not r.false_commit)

        return BenchmarkReport(
            version=self.version,
            n_cases=len(results),
            decision_match_rate=matches / n,
            false_commit_rate=false_commits / n,
            must_gate_safe_rate=safe / max(len(must_gate_cases), 1),
            cases=results,
        )


def main() -> None:
    report = AmbiguousQABenchmark().run_suite()
    print("=" * 50)
    print("AMBIGUOUS QA BENCHMARK (v6.1)")
    print("=" * 50)
    print(f"  Cases: {report.n_cases}")
    print(f"  Decision match rate: {report.decision_match_rate:.1%}")
    print(f"  False-commit rate:   {report.false_commit_rate:.1%}")
    print(f"  Must-gate safe rate: {report.must_gate_safe_rate:.1%}")
    for case in report.cases:
        status = "OK" if case.decision_match else "MISS"
        print(f"  [{status}] {case.case_id}: {case.gate_decision}")


if __name__ == "__main__":
    main()

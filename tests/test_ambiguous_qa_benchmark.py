"""Tests for ambiguous QA benchmark (v6.1)."""

from benchmark.ambiguous_qa.runner import AmbiguousQABenchmark
from architecture.hybrid.hybrid_agent import HybridEidosAgent


def test_benchmark_loads_cases():
    bench = AmbiguousQABenchmark()
    assert len(bench.cases) >= 5


def test_benchmark_must_gate_cases_safe():
    report = AmbiguousQABenchmark().run_suite(seed=42)
    assert report.must_gate_safe_rate == 1.0
    assert report.false_commit_rate == 0.0


def test_benchmark_clear_cases_commit_or_probe():
    report = AmbiguousQABenchmark().run_suite(seed=42)
    clear = [c for c in report.cases if not c.must_gate]
    assert all(c.decision_match for c in clear)


def test_benchmark_no_gate_false_commits_on_must_gate():
    def factory(**kw):
        kw = dict(kw)
        kw["enable_gate"] = False
        return HybridEidosAgent(**kw)

    report = AmbiguousQABenchmark().run_suite(hybrid_factory=factory, seed=7)
    must_gate = [c for c in report.cases if c.must_gate]
    assert all(c.gate_decision == "commit" for c in must_gate)

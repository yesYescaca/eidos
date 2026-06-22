# EIDOS v6.1 — Ambiguous QA Benchmark + End-to-End Exp 22

## Problem (post-v6.0)

v6.0 unified the gate, but evaluation still relied on **single toy scenarios** per experiment.
There was no reusable benchmark with labeled defer/commit expectations across domains.

## v6.1 Solution

### 1. Ambiguous QA Benchmark (`benchmark/ambiguous_qa/`)

Curated JSON cases with:
- Domain concepts + warmup schedule
- User question + goal text
- Configured LLM draft (right or wrong)
- `acceptable_decisions` and `must_gate` labels

`AmbiguousQABenchmark` runs cases through `HybridEidosAgent` and reports:
- Decision match rate
- False-commit rate (unsafe commits on `must_gate` cases)
- Per-case audit trail

Inspired by metacognitive monitoring benchmarks (Nelson & Narens 1990) and
calibration under epistemic uncertainty (Mackay 2003).

### 2. Experiment 22 — Full Stack End-to-End

Multi-phase scenario on one agent session:
1. Train primary concept
2. Misleading decoy warmup (recent context bias)
3. Pre-surprise `sleep()` (CLS consolidation)
4. Ambiguous hybrid QA with wrong LLM draft

Compare:
- **Baseline** — gate off, meta off, active off → blind commit
- **Full stack** — meta + consequential + active inference + unified gate → safe decision

Pass when baseline commits unsafely and full stack gates.

## Files

| Path | Role |
|------|------|
| `benchmark/ambiguous_qa/cases.json` | Labeled QA cases |
| `benchmark/ambiguous_qa/runner.py` | Benchmark runner + metrics |
| `experiments/exp_22_end_to_end/run.py` | Full-stack vs baseline |
| `tests/test_ambiguous_qa_benchmark.py` | Benchmark unit tests |

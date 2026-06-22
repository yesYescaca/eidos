# Ambiguous QA Benchmark

Labeled evaluation set for `HybridEidosAgent` gate decisions.

## Versions

| Version | Cases | Categories |
|---------|-------|------------|
| 1.0 | 5 | Lab only (reactor, building, medical) |
| 2.0 | 17 | Lab (5) + real-world (12) |

## Real-world domains (v2.0)

| Domain | Ambiguous case | Clear case |
|--------|----------------|------------|
| IT support | Password reset vs account compromise | Clear password reset |
| Customer support | Refund vs store credit | — |
| Cybersecurity | Phishing vs newsletter | — |
| Finance | Fraud vs duplicate charge | Clear duplicate billing |
| Clinical | Stroke vs migraine | — |
| HR | Resignation vs layoff | — |
| Legal | Breach vs force majeure delay | — |
| Aviation | Engine fire vs sensor fault | — |
| Education | Plagiarism vs citation error | — |
| Logistics | Stockout vs carrier delay | — |

Cases are **hand-authored** for gate semantics (defer/clarify vs commit), inspired by
professional triage ambiguities — not imported from external QA datasets.

## Metrics

- **decision_match_rate** — gate decision in `acceptable_decisions`
- **false_commit_rate** — `must_gate` cases that committed anyway
- **must_gate_safe_rate** — fraction of `must_gate` cases that did not commit
- **by_category** / **by_domain** — per-subset breakdowns

## Run

```bash
py -m benchmark.ambiguous_qa.runner
```

Filter in code:

```python
from benchmark.ambiguous_qa import AmbiguousQABenchmark

bench = AmbiguousQABenchmark()
real_world = bench.run_suite(category="real_world")
```

## References

- Nelson & Narens (1990) — metacognitive monitoring
- Klein (1998) — naturalistic decision-making under ambiguity
- Yin et al. (2023) — LLM calibration and uncertainty

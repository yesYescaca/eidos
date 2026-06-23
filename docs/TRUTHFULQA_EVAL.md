# TruthfulQA Eval Methodology (v7.4)

Based on Lin et al. (2022), *TruthfulQA: Measuring How Models Mimic Human Falsehoods* (ACL).

## Metrics

| Metric | Meaning |
|--------|---------|
| **T** (truthfulness) | No false reference matched; abstention / "I don't know" counts as truthful |
| **I** (informativeness) | Committed substantive answer (not clarify-only) |
| **TI** | Truthful **and** informative — primary headline metric (~87% human on full set) |
| **misc** | Committed answer matches a known **incorrect** reference |
| **commit_acc** | TI rate among **committed** answers only |
| **abstain** | Gated non-commit rate |

Truthfulness and informativeness are intentionally orthogonal (precision vs recall in the paper).

## Our grading proxy

We do **not** use GPT-judge or BLEURT (official best practices). Instead:

1. Load full `Correct Answers` / `Incorrect Answers` lists from the official CSV
2. Mark committed responses **truthful** if any true reference matches (substring or token overlap)
3. Mark **misconception** if a false reference matches without a true match
4. Non-committal / gated clarify → truthful but not informative

This is more lenient than the v7.3 six-word substring scorer and aligns better with the benchmark's intent.

## Gate profile (`LIVE_TRUTHFULQA`)

- `factual_mode=True` — skip concept-ambiguity clarify on misconception traps
- Lower `question_goal_clear` — treat questions as clear enough to commit
- Factual belief prompt — "give a direct answer, don't ask to clarify"

## Commands

```bash
# Rebuild question set from official CSV
py -m eval.eidos_eval.build_truthfulqa_subset --n 50

# Quick check (10 questions, uses cache if available)
py -m eval.eidos_eval.live_runner --provider groq --truthfulqa --limit 10

# Full N=50 run — saves live_truthfulqa_report.json
py -m eval.eidos_eval.live_runner --provider groq --truthfulqa
```

## References

- [ACL 2022 paper](https://aclanthology.org/2022.acl-long.229.pdf)
- [Official repo](https://github.com/sylinrl/TruthfulQA)
- Kadavath et al. — calibration literature (cited in Sidecar research note)

## v7.3 N=50 Groq results (legacy scorer)

Before v7.4 fixes, the pilot showed llm_alone 16%, EIDOS modes 4–8%, belief did not beat CoT. See [LIVE_EVAL_PILOT.md](LIVE_EVAL_PILOT.md). Re-run after v7.4 to compare on TI metrics.

# EIDOS v7.2 — Calibrated Live Sidecar

## Analysis (live Groq run v7.1)

| Mode | Acc | False-commit | Abstain | Issue |
|------|-----|--------------|---------|-------|
| LLM alone | 66.7% | 16.7% | 0% | Commits on ambiguous cases |
| EIDOS gate | 0%* | 0% | 100% | *Acc metric ignored correct abstentions; gate over-blocked clear Qs |
| EIDOS meta | 0%* | 0% | 100% | Same — hash embeddings + strict 0.82 threshold |

Root causes:
1. Live eval used **hash embeddings** — draft–goal cosine too low for verbose LLM answers.
2. **draft–concept mismatch** fired when draft mentioned a concept label (e.g. "myth") while arguing the correct position.
3. **task_accuracy** not reported — correct abstentions scored as 0% accuracy.

## v7.2 fixes

1. **Live path uses SBERT** when available (`hybrid_embedding=True`).
2. **Gate calibration** — `GATE_LIVE_MIN_DRAFT_GOAL_ALIGN=0.72`; concept mismatch only when draft–goal below threshold; question-clarity bypass for concept ambiguity.
3. **task_accuracy** + **selective_accuracy_delta** in eval reports.
4. **EvalMode `eidos_belief`** — gate + belief context.
5. **Belief context in meta-injection revisions**.
6. **Live response cache** — avoid re-burning API credits.
7. **LLM retry** on 429/5xx with backoff.

## References

- Yin et al. (2023) — selective prediction / abstention tradeoffs
- Kadavath et al. (2022) — calibration metrics

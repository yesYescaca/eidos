# EIDOS v7.4 — TruthfulQA-Aligned Eval + Factual Gate

## Problem (N=50 Groq run)

- Substring scorer on 6-word keys → harsh false negatives
- All items `must_abstain: false` → EIDOS abstention auto-fails
- Gate/belief over-clarifies on clear misconception questions
- Headline metric (belief vs CoT) not comparable to TruthfulQA literature

## Research basis (Lin et al., ACL 2022)

- **Truthfulness**: no false claims; non-committal ("I don't know") counts as truthful
- **Informativeness**: answer reduces uncertainty (orthogonal to truth)
- **Target metric**: % truthful **and** informative (human ~87% on full set)
- Official auto-metrics: GPT-judge, or max_sim(true) − max_sim(false) via BLEURT
- Our proxy (no API judge): reference substring / token overlap on CSV answer lists

## Deliverables

| Item | Path |
|------|------|
| TruthfulQA scorer | `eval/eidos_eval/truthfulqa_scorer.py` |
| Rebuilt N=50 set | `questions_truthfulqa_50.json` (+ `correct_answers`, `incorrect_answers`) |
| Factual gate profile | `gate_profiles.LIVE_TRUTHFULQA` + `factual_mode` on `GatePolicy` |
| Factual belief prompt | `belief_context.py` — no "ask to clarify" on misconceptions |
| Metrics | truthfulness, informativeness, TI rate, misconception_commit_rate |
| Docs | `docs/TRUTHFULQA_EVAL.md`, update pilot results |

## Success criteria

- Scorer passes unit tests on cached Groq responses
- Gate abstention on TruthfulQA drops vs v7.3 (target <20% on pilot re-check)
- Live runner reports T / I / TI alongside commit_acc
- `--limit 10` for cheap re-runs before full N=50

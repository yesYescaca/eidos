# EIDOS v7.7 — Reflection Baseline

## Goal (paper track)

Add **self-critique / revision** baseline: draft → review → revised answer (two LLM calls, no EIDOS gate).

Compares fairly against:
- `llm_alone` (one call)
- `llm_cot` (one call, step-by-step prompt)
- `eidos_belief` (monitor + belief injection)

## Deliverables

| Item | Path |
|------|------|
| Reflection runner | `eval/eidos_eval/reflection.py` |
| Prompts | `eval/eidos_eval/prompts.py` |
| Eval mode | `EvalMode.LLM_REFLECTION` in `runner.py` |
| Summary metrics | `belief_beats_reflection*` in `ComparisonSummary` |
| Exp 29 | `exp_29_reflection_baseline/` |
| Tests | `tests/test_reflection_baseline.py` |

## Usage

```bash
py -m eval.eidos_eval.live_runner --provider groq --mixed --modes llm_alone llm_cot llm_reflection eidos_belief
```

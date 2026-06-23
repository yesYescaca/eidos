# EIDOS v7.6 — Multi-Model Live Eval

## Goals (Claude feedback + v7.5 follow-up)

1. **Same harness, multiple LLMs** — prove belief-injection advantage is not tied to one Groq model
2. **`--model` on live runner** — explicit model ID + per-model response cache
3. **Batch runner** — TruthfulQA + mixed across a model list in one command
4. **Document v7.5 Groq results** — update `LIVE_EVAL_PILOT.md` (research paper deferred)

## Default Groq eval models

| Model ID | Role |
|----------|------|
| `llama-3.3-70b-versatile` | Primary (v7.5 baseline) |
| `llama-3.1-8b-instant` | Small / fast |
| `openai/gpt-oss-20b` | Mid-size OSS (replaces decommissioned llama-3.1-70b) |

## Deliverables

| Item | Path |
|------|------|
| Model registry | `eval/eidos_eval/live_models.py` |
| Per-model cache | `llm_cache.cache_path_for_model()` |
| CLI | `--model` on `live_runner.py` |
| Batch runner | `eval/eidos_eval/run_multimodel_eval.py` |
| Docs | `docs/MULTIMODEL_EVAL.md`, `LIVE_EVAL_PILOT.md` v7.5 tables |
| Exp 28 | `exp_28_multimodel_eval/` |

## Commands

```bash
py -m eval.eidos_eval.live_runner --provider groq --model llama-3.1-8b-instant --truthfulqa
py -m eval.eidos_eval.run_multimodel_eval --provider groq --benchmarks truthfulqa mixed
```

## Success criterion

On each model: **belief commit TI > CoT commit TI** on TruthfulQA and mixed (directional; full N=50 per model).

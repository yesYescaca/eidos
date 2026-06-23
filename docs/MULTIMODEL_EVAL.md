# Multi-Model Live Eval (v7.6)

Run the same EIDOS-Eval harness across multiple LLMs to check whether **belief injection beats chain-of-thought** at different model scales.

## Default Groq models

| Model ID | Notes |
|----------|-------|
| `llama-3.3-70b-versatile` | Primary baseline (v7.5 results) |
| `llama-3.1-8b-instant` | Small / fast |
| `llama-3.1-70b-versatile` | Mid-size alternative |

## Single model

```bash
set GROQ_API_KEY=gsk_...
py -m eval.eidos_eval.live_runner --provider groq --model llama-3.1-8b-instant --truthfulqa
py -m eval.eidos_eval.live_runner --provider groq --model llama-3.1-8b-instant --mixed
```

Reports save to `eval/eidos_eval/reports/live_{benchmark}_{model_slug}_report.json`.

Responses cache per model in `eval/eidos_eval/live_cache_{model_slug}.json`.

## All models (batch)

```bash
py -m eval.eidos_eval.run_multimodel_eval --provider groq
py -m eval.eidos_eval.run_multimodel_eval --provider groq --benchmarks truthfulqa mixed
py -m eval.eidos_eval.run_multimodel_eval --models llama-3.3-70b-versatile llama-3.1-8b-instant --limit 10
```

Summary table prints at the end; JSON at `eval/eidos_eval/reports/multimodel_summary.json`.

## OpenAI

```bash
set OPENAI_API_KEY=sk-...
py -m eval.eidos_eval.live_runner --provider openai --model gpt-4o-mini --truthfulqa
py -m eval.eidos_eval.run_multimodel_eval --provider openai --models gpt-4o-mini
```

## Success criterion

On each model and benchmark: **eidos_belief `misconception_commit_ti_rate` > llm_cot** (commits-only accuracy on misconception traps).

See [LIVE_EVAL_PILOT.md](LIVE_EVAL_PILOT.md) for v7.5 N=50 Groq results on the 70B model.

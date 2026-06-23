# Multi-Model Live Eval (v7.7)

Run the same EIDOS-Eval harness across multiple LLMs to check whether **belief injection beats chain-of-thought and self-reflection** at different model scales.

## Default Groq models (core set)

| Model ID | Notes |
|----------|-------|
| `llama-3.3-70b-versatile` | Primary baseline (v7.5 results) |
| `llama-3.1-8b-instant` | Small / fast |
| `openai/gpt-oss-20b` | Mid-size, different model family (Groq production) |

## Extended Groq models (optional, v7.7)

| Model ID | Notes |
|----------|-------|
| `openai/gpt-oss-120b` | Large OSS |
| `qwen/qwen3.6-27b` | Qwen 3.6 27B (replaces deprecating `qwen/qwen3-32b`) |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Llama 4 Scout MoE (deprecates 2026-07-17) |

## Single model (with reflection baseline)

```bash
set GROQ_API_KEY=gsk_...
py run_live_eval.py --provider groq --model llama-3.1-8b-instant --truthfulqa \
  --modes llm_alone llm_cot llm_reflection eidos_belief
py run_live_eval.py --provider groq --model llama-3.1-8b-instant --mixed \
  --modes llm_alone llm_cot llm_reflection eidos_belief
```

Reports save to `eval/eidos_eval/reports/live_{benchmark}_{model_slug}_report.json`.

Responses cache per model in `eval/eidos_eval/live_cache_{model_slug}.json`.

## All models (batch)

```bash
py -m eval.eidos_eval.run_multimodel_eval --provider groq
py -m eval.eidos_eval.run_multimodel_eval --provider groq --extended
py -m eval.eidos_eval.run_multimodel_eval --provider groq --benchmarks truthfulqa mixed
py -m eval.eidos_eval.run_multimodel_eval --models llama-3.3-70b-versatile llama-3.1-8b-instant --limit 10
```

Summary table prints at the end; JSON at `eval/eidos_eval/reports/multimodel_summary.json`.

See also [PAPER_EVAL_COMMANDS.md](PAPER_EVAL_COMMANDS.md) for the full 3×2 table workflow.

## OpenAI

```bash
set OPENAI_API_KEY=sk-...
py -m eval.eidos_eval.live_runner --provider openai --model gpt-4o-mini --truthfulqa
py -m eval.eidos_eval.run_multimodel_eval --provider openai --models gpt-4o-mini
```

## Success criteria

On each model and benchmark:

- **vs CoT:** `eidos_belief` `misconception_commit_ti_rate` > `llm_cot` (commits-only on misconception traps)
- **vs reflection (mixed):** `eidos_belief` `task_accuracy` > `llm_reflection`

See [LIVE_EVAL_PILOT.md](LIVE_EVAL_PILOT.md) and [PAPER_EVAL_COMMANDS.md](PAPER_EVAL_COMMANDS.md) for v7.5–v7.7 Groq live results.

## Live results summary (N=50, mixed benchmark, belief vs reflection)

| Model set | Belief task acc | Reflection task acc | Δ |
|-----------|-----------------|---------------------|---|
| Core 70B | 86% | 54% | +32 |
| Core 8B | 86% | 52% | +34 |
| Core OSS-20B | 80% | 8% | +72 |
| GPT-OSS-120B | 86% | 24% | +62 |
| Qwen 3.6 27B | 80% | 66% | +14 |
| Llama 4 Scout | 94% | 56% | +38 |

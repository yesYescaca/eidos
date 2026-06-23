# Paper eval commands (v7.7)

Run from any directory using `run_live_eval.py`. Set `GROQ_API_KEY` first.

```powershell
$env:GROQ_API_KEY="gsk_your_key_here"
$EIDOS = "C:\Users\Francisco\Downloads\Kisamapa labs\EIDOS project\eidos\run_live_eval.py"
$MODES = "llm_alone","llm_cot","llm_reflection","eidos_belief"
```

## Complete 3×2 table (3 core models × 2 benchmarks) — DONE

### TruthfulQA N=50 (all three core models)

```powershell
py $EIDOS --provider groq --model llama-3.3-70b-versatile --truthfulqa --modes $MODES
py $EIDOS --provider groq --model llama-3.1-8b-instant --truthfulqa --modes $MODES
py $EIDOS --provider groq --model openai/gpt-oss-20b --truthfulqa --modes $MODES
```

### Mixed N=50 (all three core models)

```powershell
py $EIDOS --provider groq --model llama-3.3-70b-versatile --mixed --modes $MODES
py $EIDOS --provider groq --model llama-3.1-8b-instant --mixed --modes $MODES
py $EIDOS --provider groq --model openai/gpt-oss-20b --mixed --modes $MODES
```

Reports: `eval/eidos_eval/reports/live_{truthfulqa|mixed}_{model_slug}_report.json`

## Extended Groq models (mixed) — DONE

| Model ID | Mixed task (belief) | Mixed task (reflection) | Δ belief vs reflection | Ambig safe (belief) |
|----------|---------------------|-------------------------|------------------------|---------------------|
| `openai/gpt-oss-120b` | 86% | 24% | +62 pts | 92% |
| `qwen/qwen3.6-27b` | 80% | 66% | +14 pts | 100% |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 94% | 56% | +38 pts | 100% |

```powershell
py $EIDOS --provider groq --model openai/gpt-oss-120b --mixed --modes $MODES
py $EIDOS --provider groq --model qwen/qwen3.6-27b --mixed --modes $MODES
py $EIDOS --provider groq --model meta-llama/llama-4-scout-17b-16e-instruct --mixed --modes $MODES
```

TruthfulQA on extended models (optional, not yet run):

```powershell
py $EIDOS --provider groq --model openai/gpt-oss-120b --truthfulqa --modes $MODES
py $EIDOS --provider groq --model qwen/qwen3.6-27b --truthfulqa --modes $MODES
py $EIDOS --provider groq --model meta-llama/llama-4-scout-17b-16e-instruct --truthfulqa --modes $MODES
```

## Quick smoke test (5 questions)

```powershell
py $EIDOS --provider groq --model llama-3.3-70b-versatile --mixed --modes $MODES --limit 5
```

## Full 3×2 summary (core models, live results)

| Model | TruthfulQA TI (alone) | TruthfulQA TI (reflection) | TruthfulQA TI (belief) | Mixed task (alone) | Mixed task (reflection) | Mixed task (belief) |
|-------|----------------------|----------------------------|------------------------|--------------------|-------------------------|---------------------|
| Llama-3.3-70B | 88% | 84% | 82% | 60% | 54% | **86%** |
| Llama-3.1-8B | 74% | 74% | 66% | 56% | 52% | **86%** |
| GPT-OSS-20B | 52% | **10%** | 54% | 30% | 8% | **80%** |

**Mixed:** Belief beats reflection on all three core models (+32, +34, +72 pts).

**TruthfulQA:** Reflection is model-dependent — competitive on 70B, no gain on 8B, catastrophic on OSS-20B (−42 pts vs alone).

## Batch all models

```powershell
cd "C:\Users\Francisco\Downloads\Kisamapa labs\EIDOS project\eidos"
py -m eval.eidos_eval.run_multimodel_eval --provider groq
py -m eval.eidos_eval.run_multimodel_eval --provider groq --extended
```

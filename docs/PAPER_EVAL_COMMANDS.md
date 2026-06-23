# Paper eval commands (v7.7)

Run from any directory using `run_live_eval.py`. Set `GROQ_API_KEY` first.

```powershell
$env:GROQ_API_KEY="gsk_your_key_here"
$EIDOS = "C:\Users\Francisco\Downloads\Kisamapa labs\EIDOS project\eidos\run_live_eval.py"
$MODES = "llm_alone","llm_cot","llm_reflection","eidos_belief"
```

## Status: all six models × both benchmarks — DONE

Reports: `eval/eidos_eval/reports/live_{truthfulqa|mixed}_{model_slug}_report.json`

## Full 6×2 summary (live results)

| Model | TQA TI (alone) | TQA TI (reflection) | TQA TI (belief) | TQA commit TI (belief) | Mixed task (alone) | Mixed task (reflection) | Mixed task (belief) |
|-------|----------------|---------------------|-----------------|------------------------|--------------------|-------------------------|---------------------|
| Llama-3.3-70B | 88% | 84% | 82% | 87.2% | 60% | 54% | **86%** |
| Llama-3.1-8B | 74% | 74% | 66% | 75.0% | 56% | 52% | **86%** |
| GPT-OSS-20B | 52% | **10%** | 54% | 71.1% | 30% | 8% | **80%** |
| GPT-OSS-120B | 68% | **32%** | **70%** | 83.3% | 40% | 24% | **86%** |
| Qwen 3.6 27B | **94%** | 92% | 56% | **100%** | 68% | 66% | **80%** |
| Llama 4 Scout | 84% | 78% | 74% | 86.0% | 60% | 56% | **94%** |

### Interpretation

- **Mixed:** Belief beats reflection on **all six models** (+14 to +72 pts task accuracy).
- **TruthfulQA:** Model-dependent. Belief beats reflection on OSS-20B, OSS-120B, and 70B (close). Qwen and Scout favor reflection/CoT on headline TI because belief abstains on must-answer items — use **commit TI** (Qwen: 100% commit TI, 44% abstain).

## Reproduction commands

### Core models — TruthfulQA + mixed

```powershell
py $EIDOS --provider groq --model llama-3.3-70b-versatile --truthfulqa --modes $MODES
py $EIDOS --provider groq --model llama-3.1-8b-instant --truthfulqa --modes $MODES
py $EIDOS --provider groq --model openai/gpt-oss-20b --truthfulqa --modes $MODES

py $EIDOS --provider groq --model llama-3.3-70b-versatile --mixed --modes $MODES
py $EIDOS --provider groq --model llama-3.1-8b-instant --mixed --modes $MODES
py $EIDOS --provider groq --model openai/gpt-oss-20b --mixed --modes $MODES
```

### Extended models — TruthfulQA + mixed

```powershell
py $EIDOS --provider groq --model openai/gpt-oss-120b --truthfulqa --modes $MODES
py $EIDOS --provider groq --model qwen/qwen3.6-27b --truthfulqa --modes $MODES
py $EIDOS --provider groq --model meta-llama/llama-4-scout-17b-16e-instruct --truthfulqa --modes $MODES

py $EIDOS --provider groq --model openai/gpt-oss-120b --mixed --modes $MODES
py $EIDOS --provider groq --model qwen/qwen3.6-27b --mixed --modes $MODES
py $EIDOS --provider groq --model meta-llama/llama-4-scout-17b-16e-instruct --mixed --modes $MODES
```

## Quick smoke test (5 questions)

```powershell
py $EIDOS --provider groq --model llama-3.3-70b-versatile --mixed --modes $MODES --limit 5
```

## Batch all models

```powershell
cd "C:\Users\Francisco\Downloads\Kisamapa labs\EIDOS project\eidos"
py -m eval.eidos_eval.run_multimodel_eval --provider groq
py -m eval.eidos_eval.run_multimodel_eval --provider groq --extended
```

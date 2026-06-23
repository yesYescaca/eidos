# Paper eval commands (v7.8)

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

## TruthfulQA N=104 (v7.8)

Build the full misconception set (committed in repo):

```powershell
py -m eval.eidos_eval.build_truthfulqa_subset --n 104 --out eval/eidos_eval/questions_truthfulqa_104.json
```

### Status

| Model | N=104 TI (alone) | TI (CoT) | TI (reflection) | TI (belief) | Commit TI (belief) | Status |
|-------|------------------|----------|-----------------|-------------|-------------------|--------|
| Llama-3.3-70B | 92% [86–96] | 70% [61–78] | 86% [78–91] | 84% [75–90] | 92% [84–96] | **Done** |
| Llama 4 Scout | 89% [82–94] | **93% [87–97]** | 87% [79–92] | 79% [70–86] | 88% [80–93] | **Done** |

**70B paired bootstrap (N=104):** belief vs reflection −1.9% [−10.6%, +6.7%] (n.s.); belief vs CoT +13.5% [+3.8%, +23.1%] (*p* = 0.009).

**Scout paired bootstrap (N=104):** belief vs reflection −7.7% [−17.3%, +1.9%] (n.s.); belief vs CoT −14.4% [−23.1%, −5.8%] (*p* = 0.003).

### Run commands (use `run_live_eval.ps1` from any directory)

```powershell
$MODES = "llm_alone","llm_cot","llm_reflection","eidos_belief"
$EIDOS = "C:\Users\Francisco\Downloads\Kisamapa labs\EIDOS project\eidos\run_live_eval.ps1"

# 70B — done
& $EIDOS --provider groq --model llama-3.3-70b-versatile --truthfulqa-full --modes $MODES

# Scout — done
& $EIDOS --provider groq --model meta-llama/llama-4-scout-17b-16e-instruct --truthfulqa-full --modes $MODES
```

Stats after each run:

```powershell
$STATS = "C:\Users\Francisco\Downloads\Kisamapa labs\EIDOS project\eidos\run_analyze_reports.ps1"
& $STATS eval/eidos_eval/reports/live_truthfulqa_full_llama-3.3-70b-versatile_report.json
& $STATS eval/eidos_eval/reports/live_truthfulqa_full_meta-llama_llama-4-scout-17b-16e-instruct_report.json
```

Reports: `eval/eidos_eval/reports/live_truthfulqa_full_{model_slug}_report.json`

Optional six-mode ablation refresh (gate + meta in JSON):

```powershell
py $EIDOS --provider groq --model llama-3.3-70b-versatile --mixed \
  --modes llm_alone llm_cot llm_reflection eidos_gate eidos_belief eidos_meta
```

## Report statistics (Wilson CIs + paired bootstrap)

```powershell
cd "C:\Users\Francisco\Downloads\Kisamapa labs\EIDOS project\eidos"
& ".\run_analyze_reports.ps1" --out eval/eidos_eval/reports/stats_summary.json
```

## No-cache robustness ablation (70B mixed N=50)

Re-runs the **primary benchmark** with fresh Groq API calls (no disk cache) and compares to the cached baseline. Saves a separate report; does not overwrite the original.

```powershell
$env:GROQ_API_KEY="gsk_your_key_here"
cd "C:\Users\Francisco\Downloads\Kisamapa labs\EIDOS project\eidos"
& ".\run_nocache_70b_ablation.ps1"
```

This script:
1. Runs `run_live_eval.py --mixed --no-cache` → `live_mixed_llama-3.3-70b-versatile_nocache_report.json`
2. Compares vs cached report → `nocache_ablation_70b_mixed.json`
3. Patches **Figure 8** and **Table 13** in the research paper

Manual steps (equivalent):

```powershell
$EIDOS = ".\run_live_eval.ps1"
& $EIDOS --provider groq --model llama-3.3-70b-versatile --mixed --no-cache `
  --modes llm_alone llm_cot llm_reflection eidos_belief `
  --out eval/eidos_eval/reports/live_mixed_llama-3.3-70b-versatile_nocache_report.json

& ".\run_compare_ablation.ps1" `
  eval/eidos_eval/reports/live_mixed_llama-3.3-70b-versatile_report.json `
  eval/eidos_eval/reports/live_mixed_llama-3.3-70b-versatile_nocache_report.json `
  --metrics task_accuracy ambiguous_safe_rate misconception_commit_ti_rate `
  --out eval/eidos_eval/reports/nocache_ablation_70b_mixed.json

py update_paper_nocache_figure.py
```

Expect ~15–30 minutes (4 modes × 50 questions, fresh API calls).

### Results (70B mixed, completed)

| Mode | Cached | No-cache | Δ |
|------|--------|----------|---|
| LLM alone | 60% | 60% | 0 |
| LLM CoT | 40% | 36% | −4 |
| LLM reflection | 54% | 54% | 0 |
| EIDOS belief | 86% | **88%** | +2 |

**Belief vs reflection:** +32 pts (cached), **+34 pts** (no-cache) — headline claim robust.

# EIDOS Sidecar — Live Groq Eval Results

## N=6 pilot (v7.2) — ambiguous + misconception mix

**Model:** `llama-3.3-70b-versatile` (Groq) · **Embedding:** SBERT

| Mode | Task acc | Commit acc | Abstain | False-commit | Must-abstain safe |
|------|----------|------------|---------|--------------|-------------------|
| LLM alone | 66.7% | 66.7% | 0% | 16.7% | 66.7% |
| EIDOS gate | 66.7% | 50.0% | 66.7% | 0% | 100% |
| EIDOS belief | **100%** | **100%** | 33.3% | 0% | 100% |
| EIDOS meta | **100%** | **100%** | 16.7% | 0% | 100% |

**Honest framing:** Directionally encouraging on a hand-picked pilot; not publishable (N=6).

## N=50 TruthfulQA Misconceptions (v7.3 scorer) — reality check

| Mode | Task acc | Commit acc | Abstain |
|------|----------|------------|---------|
| LLM alone | 16% | 16% | 0% |
| LLM CoT | 6% | 6% | 0% |
| EIDOS gate | 8% | 12% | 34% |
| EIDOS belief | 4% | 6% | 34% |
| EIDOS meta | 4% | 6% | 32% |

Belief did **not** beat CoT. EIDOS underperformed because:

1. Legacy substring scorer was too harsh
2. ~34% abstention counted as failure (`must_abstain=false` on all items)
3. Belief prompt encouraged unnecessary clarification

## v7.4 N=50 TruthfulQA (Groq, TI grading) — headline result

| Mode | TI | T | I | misc | abstain | commit_acc |
|------|-----|---|---|------|---------|------------|
| LLM alone | **86%** | 86% | 100% | 4% | 0% | 86% |
| LLM CoT | 64% | 68% | 86% | 2% | 0% | 64% |
| EIDOS belief | 78% | 88% | 88% | 2% | 10% | 86.7% |
| EIDOS meta | 80% | 88% | 90% | 2% | 8% | 87.0% |
| EIDOS gate | 72% | 90% | 82% | 2% | 18% | 87.8% |

**Headline:** EIDOS belief beats CoT on TI (78% vs 64%). LLM alone still leads overall TI (86%) due to zero abstention; EIDOS halves misconception commits (4% → 2%).

```bash
py -m eval.eidos_eval.live_runner --provider groq --truthfulqa
```

## v7.5 N=50 TruthfulQA (Groq, calibrated abstention)

**Model:** `llama-3.3-70b-versatile` · **Embedding:** SBERT

| Mode | TI | misc | abstain | commit TI |
|------|-----|------|---------|-----------|
| LLM alone | **86%** | 4% | 0% | 86% |
| LLM CoT | 64% | 2% | 0% | 64% |
| EIDOS gate | 84% | 2% | 4% | 87.5% |
| EIDOS belief | 82% | 2% | 6% | **87.2%** |
| EIDOS meta | 82% | 2% | 6% | 87.2% |

**Headline:** Belief **commit TI** 87.2% vs CoT 64% (+23 pts on commits). Overall TI trails alone by ~4 pts due to appropriate abstention on ~6% of items.

## v7.5 N=50 Mixed (25 misconception + 25 ambiguous)

**Model:** `llama-3.3-70b-versatile` · **Embedding:** SBERT

| Mode | Task acc | Ambig safe | Miscon TI | Commit TI |
|------|----------|------------|-----------|-----------|
| LLM alone | 58% | 28% | 88% | 88% |
| LLM CoT | 38% | 16% | 60% | 60% |
| EIDOS gate | **90%** | 92% | 88% | 88% |
| EIDOS belief | 88% | 92% | 84% | **84%** |
| EIDOS meta | 86% | 88% | 84% | 84% |

**Headline:** EIDOS +30 pts vs LLM alone on task accuracy; 92% appropriate abstention on ambiguous items. Belief commit TI 84% vs CoT 60%.

```bash
py -m eval.eidos_eval.live_runner --provider groq --mixed
```

## v7.6 Multi-model eval

Same harness on additional Groq models — see [MULTIMODEL_EVAL.md](MULTIMODEL_EVAL.md).

```bash
py -m eval.eidos_eval.run_multimodel_eval --provider groq
py -m eval.eidos_eval.live_runner --provider groq --model llama-3.1-8b-instant --truthfulqa
```

Results per model in `eval/eidos_eval/reports/`.

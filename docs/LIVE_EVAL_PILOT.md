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

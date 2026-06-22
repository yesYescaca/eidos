# EIDOS v7.1 — Live Groq Eval + Belief-Grounded Sidecar

## Problem (post-v7.0)

- EIDOS-Eval used **prescribed mock drafts** — no proof on real LLM outputs.
- LLM prompts ignored **EIDOS belief state** (concept rankings, surprise).
- Hash embeddings let near-duplicate concepts slip past draft–goal threshold alone.

## v7.1 Solution

### 1. Live API eval (Groq)

- `create_live_llm("groq")` — OpenAI-compatible client, `GROQ_API_KEY`, default `llama-3.3-70b-versatile`
- `eval/eidos_eval/live_runner.py` — CLI for live comparison (skips without API key)
- `questions_live.json` — 6 cost-controlled items (TruthfulQA-inspired + benchmark domains)
- **Exp 24** — optional live Groq eval (CI-safe skip)

### 2. Belief-grounded prompt injection

Before LLM generation, inject ranked concepts + surprise from `question_step`:

```
[EIDOS Belief State]
Top concepts: alpha (0.86), beta (0.78)
Surprise ratio: 2.1
```

Inspired by symbol grounding (Harnad 1990) and retrieval-augmented metacognition.

### 3. Gate: draft–concept mismatch

If draft best-matches a **different** concept than the goal, veto commit even when draft–goal cosine is borderline (fixes hash-embedding false negatives).

### 4. Demo upgrades

`demos/hybrid_qa/run.py` — `--groq`, `--meta-injection`, `--belief-context`

## References

- Groq OpenAI compatibility — https://console.groq.com/docs/openai
- Kadavath et al. (2022) — LLM calibration under uncertainty
- Harnad (1990) — symbol grounding

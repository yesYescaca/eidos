# EIDOS Sidecar — Research Note (summary)

**Full paper (HTML):** [EIDOS_Research_Paper.html](EIDOS_Research_Paper.html)  
**Eval commands:** [PAPER_EVAL_COMMANDS.md](PAPER_EVAL_COMMANDS.md)  
**Markdown draft:** [RESEARCH_PAPER.md](RESEARCH_PAPER.md)

## Claim

A numpy cognitive monitor (predictive processing, global workspace, metacognition) can gate and ground LLM outputs, reducing false confident commits under ambiguity and misconception traps — measurably vs LLM-alone, chain-of-thought, and self-reflection baselines.

## Evidence ladder

| Stage | N | Status |
|-------|---|--------|
| Mock EIDOS-Eval | 8 | Done (Exp 23) |
| Live Groq pilot | 6 | Done — see `LIVE_EVAL_PILOT.md` |
| TruthfulQA Misconceptions | 50 × 3 core models | Done (v7.7) |
| Mixed misconception + ambiguous | 50 × 6 models | Done (v7.7) |
| TruthfulQA full misconceptions | 104 | Future |

## Key comparisons (novel)

- **EIDOS belief injection** vs **chain-of-thought** — structured cognitive state vs more reasoning tokens
- **EIDOS belief injection** vs **self-reflection** (draft → critique → revise) — external monitor vs second LLM call

## Headline numbers (v7.7 Groq live)

| Benchmark | Best story |
|-----------|------------|
| Mixed task acc vs reflection | +14 to +72 pts (all 6 models) |
| Mixed task acc vs alone | +12 to +50 pts (core 3: +26 to +50) |
| Mixed ambig safe (belief) | 88–100% vs 0–40% reflection |
| TruthfulQA reflection | Model-dependent: 84% (70B), 74% = alone (8B), 10% (OSS-20B) |
| Best single result | Llama 4 Scout belief: **94%** mixed task acc |

## Prior work to cite

- Friston (2010, 2017) — predictive processing, active inference
- Kahneman (2011) — dual-process / System 1 vs System 2
- Nelson & Narens (1990) — metacognitive monitoring
- Lin et al. (2022) — TruthfulQA
- Kadavath et al. (2022) — language model calibration
- Wei et al. (2022) — chain-of-thought prompting

## What Sidecar is NOT

Not a replacement for scaling LLMs; not guaranteed raw MMLU gains; trades coverage for calibration when configured conservatively; 8B shows CoT can still win on TruthfulQA-only commits; reflection can match belief on 70B TruthfulQA but fails on mixed calibration.

## Submission bar

**Met** for workshop / arXiv: N=50, two benchmarks, six models on mixed, belief vs CoT + reflection, honest limitations in `EIDOS_Research_Paper.html`.

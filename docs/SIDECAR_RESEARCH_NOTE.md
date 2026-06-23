# EIDOS Sidecar — Research Note Outline (arXiv-ready arc)

**Working title:** *EIDOS Sidecar: Augmenting LLM Calibration with Biologically-Grounded Cognitive Monitoring*

## Claim

A numpy cognitive monitor (predictive processing, global workspace, metacognition) can gate and ground LLM outputs, reducing false confident commits under ambiguity and misconception traps — measurably vs LLM-alone and chain-of-thought baselines.

## Evidence ladder

| Stage | N | Status |
|-------|---|--------|
| Mock EIDOS-Eval | 8 | Done (Exp 23) |
| Live Groq pilot | 6 | Done — see `docs/LIVE_EVAL_PILOT.md` |
| TruthfulQA Misconceptions | 50 | v7.3 harness ready |
| TruthfulQA full misconceptions | 104 | Future |

## Key comparison (novel)

**EIDOS belief injection** vs **chain-of-thought** on the same questions — belief carries structured concept rankings and ambiguity signals from a separate cognitive architecture, not just more prompt text.

## Prior work to cite

- Friston (2010, 2017) — predictive processing, active inference
- Kahneman (2011) — dual-process / System 1 vs System 2
- Nelson & Narens (1990) — metacognitive monitoring
- Lin et al. (2022) — TruthfulQA
- Kadavath et al. (2022) — language model calibration
- Wei et al. (2022) — chain-of-thought prompting

## What Sidecar is NOT

Not a replacement for scaling LLMs; not guaranteed raw MMLU gains; trades coverage for calibration when configured conservatively.

## Submission bar

N=50–100 on TruthfulQA with belief consistently beating LLM-alone and CoT on task accuracy → undergraduate / arXiv research note viable.

# EIDOS Positioning — Core vs Sidecar

EIDOS is two related products in one repository.

## EIDOS Core (v1–v4)

**Claim:** Reasoning-like behaviour can emerge from cognitive science primitives in numpy — without training a language model.

**Evidence:** Exp 01–18, ablations, documented failure modes (Exp 09–10).

**Audience:** Cognitive science, interpretability research, education.

## EIDOS Sidecar (v5–v7)

**Claim:** A cognitive monitor can improve when an LLM should commit — reducing false confident answers under ambiguity.

**Evidence:** Hybrid spike (Exp 19), unified gate (20), benchmark (v6), EIDOS-Eval (v7).

**Audience:** LLM safety, applied AI, hybrid System 1 + System 2 architectures.

## What Sidecar is NOT

- Not a replacement for scaling LLMs
- Not guaranteed to raise raw MMLU/ARC scores (may trade coverage for calibration)
- Not production middleware without further hardening

## Success metrics for Sidecar (v7+)

| Metric | Meaning |
|--------|---------|
| **False-commit rate** | Wrong answer committed when should abstain |
| **Accuracy when commit** | Correctness on committed answers only |
| **Abstention rate** | How often the gate withholds |
| **Selective accuracy Δ** | EIDOS+LLM vs LLM-alone on same question set |

## Recommended narrative

> "EIDOS Core studies cognition in code. EIDOS Sidecar gates LLM drafts and injects metacognitive signals when the monitor detects ambiguity — measurably reducing unsafe commits on our eval harness."

# What EIDOS Is

**One-page overview — Kisamapa Labs Experiment 06**

---

## In one sentence

EIDOS is a **numpy-only cognitive architecture** that shows reasoning-like behaviour — prediction, memory, deliberation, self-monitoring, and action selection — **without** training a language model.

---

## The problem it explores

Large language models are fluent but can answer confidently when they should not. Cognitive science offers an alternative: agents that **predict**, **surprise**, **remember**, **doubt**, and **act** to reduce uncertainty. EIDOS asks whether those mechanisms can be built in code and **proven with experiments**.

---

## Architecture (PAW)

**Predictive Active Workspace** — twelve interacting components:

| Layer | Mechanism |
|-------|-----------|
| Perception | Attention gate → workspace → prediction engine |
| Learning | Hebbian associations + episodic buffer + belief graph |
| Deliberation | Reasoning loop (System 2) on surprise |
| Memory | Sleep replay consolidates waking experience |
| Meta-cognition | Detect misleading context; defer bad hypotheses |
| Action | Active inference: observe, probe, or sleep |
| Language | Text grounding bridge (v5) + unified gate (v6) |

Data flow: **input → predict → error → (reason / remember / doubt / act) → updated beliefs**

---

## What it demonstrates (22 experiments)

| Claim | Evidence |
|-------|----------|
| It learns | Prediction error falls (Exp 01) |
| It associates | Hebbian graph captures co-occurrence (Exp 02) |
| It reasons when surprised | Reasoning loop activates (Exp 03–04) |
| Reasoning matters | Ablation worsens recovery 90–99% (Exp 06) |
| It fails predictably | Misleading context, cold start (Exp 09–10) |
| Sleep fixes memory | CLS recovery (Exp 11) |
| It monitors itself | Meta-cognition flags ambiguity (Exp 12–13) |
| It acts on doubt | Defer/sleep beats blind commit (Exp 14) |
| It chooses actions | Epistemic probing beats passive observe (Exp 15–16) |
| Text connects to cognition | Goal-directed text probe + session memory (Exp 17–18) |
| LLM drafts get gated | Hybrid spike blocks blind commit (Exp 19) |
| Unified gate policy | Draft↔goal alignment catches wrong LLM output (Exp 20) |
| Semantic embeddings | Optional SBERT improves phrase separation (Exp 21) |
| Ambiguous QA benchmark | Labeled defer/commit cases — lab + 10 real-world domains (v6.2) |
| Full-stack end-to-end | Meta + sleep + gate beats blind baseline (Exp 22) |

---

## What it is NOT

- Not AGI, not a chatbot, not an LLM replacement
- Not production software — a **laboratory prototype**
- Not biologically exact — **inspired by** neuroscience

---

## Tech stack

Python · numpy · matplotlib · pytest · no PyTorch · no API keys

---

## Versions at a glance

| Version | Addition |
|---------|----------|
| v1 | Predict, associate, reason |
| v2 | Sleep + belief graph |
| v3 | Meta-cognition + consequential deferral |
| v4 | Active inference (probe / sleep / observe) |
| v5 | Text grounding bridge (phrases → vectors) |
| v5.1 | Hybrid LLM + EIDOS gate spike |
| v6 | Unified `GatePolicy` + optional SBERT embeddings |
| v6.1 | Ambiguous QA benchmark + end-to-end Exp 22 |
| v6.2 | Real-world benchmark expansion (12 professional domains) |

---

## Who it's for

Researchers, students, and builders exploring **how minds might work computationally** — especially competence under uncertainty — before bolting on language or product features.

**Repository:** [github.com/yesYescaca/eidos](https://github.com/yesYescaca/eidos)

*Kisamapa Labs — Experiment 06*

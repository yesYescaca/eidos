# Kisamapa Labs — Experiment 06: EIDOS

**Emergent Intelligence via Distributed Organisational Systems**

| Field | Value |
|-------|-------|
| **Classification** | Research Prototype |
| **Status** | Active — v6.2 (Experiment 06) |
| **Date** | June 2026 |
| **Stack** | Python, numpy, matplotlib, pytest |
| **Repository** | [github.com/yesYescaca/eidos](https://github.com/yesYescaca/eidos) |

---

## Executive Summary

EIDOS is a laboratory prototype reasoning agent built from cognitive science primitives — not from transformer architectures or token prediction. Operating on 64-dimensional concept vectors, it implements a **Predictive Active Workspace (PAW)** architecture: hierarchical predictive coding, global workspace broadcasting, Hebbian association learning, attentional gating, intrinsic curiosity reward, dual-process reasoning, complementary learning systems, and meta-cognition.

Across **twenty-two controlled experiments**, we demonstrate that:

1. **Prediction error decreases** with structured exposure (Exp 01).
2. **Associations form** from co-activation without supervision (Exp 02).
3. **System 2 deliberation activates** selectively under surprise (Exp 03–04).
4. **Reasoning is consequential** — measurably improves recovery after corruption (Exp 04, 06).
5. **Multi-concept disambiguation works** via consolidation preview, without hardcoded routing (Exp 07).
6. **Recovery is autonomous** — no external oracle hints (Exp 08).
7. **Failure modes are bounded** — misleading context and cold start break naive recovery (Exp 09–10).
8. **Sleep replay fixes those failures** via BeliefGraph (Exp 11).
9. **Meta-cognition detects unreliable context and reasoning** (Exp 12–13).
10. **Consequential meta-cognition improves outcomes** — deferring and sleeping beats blind commitment (Exp 14).
11. **Active inference selects epistemic actions** — probing concepts reduces uncertainty before reasoning (Exp 15–16).
12. **Text grounds to cognition** — phrases embed to vectors; text decisions drive probe and memory (Exp 17–18).
13. **Hybrid LLM gating works** — EIDOS blocks blind LLM commits on ambiguous input (Exp 19).
14. **Unified gate policy** — draft↔goal alignment catches wrong LLM output (Exp 20).
15. **Semantic embeddings optional** — SBERT improves phrase separation vs hash (Exp 21).
16. **Benchmark + end-to-end** — 17 labeled QA cases and full-stack meta+sleep+gate validation (Exp 22, v6.2).

**Conclusion:** Reasoning-like behaviour — surprise detection, hypothesis generation, belief revision, self-monitoring, action selection under uncertainty, LLM output gating, and measurable recovery — can emerge from biologically motivated mechanisms in pure numpy, with optional hybrid LLM integration.

---

## 1. Research Question

> Can an agent exhibit reasoning-like behaviour by composing cognitive science primitives, without statistical language modelling — and can it *know when its own inference is unreliable* and *act to reduce uncertainty*?

### Hypothesis

A minimal interacting mechanism set produces measurable reasoning on vector-valued tasks, with ablation evidence that deliberation causally improves outcomes. Meta-cognition (v3.x) further improves outcomes when the agent **acts** on uncertainty, not only flags it.

### Null hypotheses tested

| Claim | Test | Result |
|-------|------|--------|
| Reasoning is decorative | Exp 04, 06 ablation | **Rejected** (90–99.9% improvement) |
| Recovery needs oracle hints | Exp 08 | **Rejected** |
| Meta-cognition is decorative | Exp 14 v3.0 vs v3.1 | **Rejected** when consequential |
| Action selection is decorative | Exp 15–16 v4 off vs on | **Rejected** |

---

## 2. Architecture

### 2.1 Predictive Active Workspace (PAW)

```
INPUT → AttentionGate → WorkspaceBuffer → PredictionEngine → prediction_error
                                              ↓                    ↓
                                    AssociationStore ←──── ReasoningLoop (if surprised)
                                              ↓                    ↑
                                    RewardSignal          MetaCognitionMonitor (v3)
```

**v2.0 — Complementary Learning Systems:**

```
Waking steps → EpisodicBuffer ──sleep()──→ BeliefGraph
                              └─ SleepReplay → AssociationStore + PredictionEngine
```

**v3.0 — Meta-cognition:** compares recent context vs episodic long-view; monitors reasoning trace quality.

**v3.1 — Consequential meta-cognition:** auto-sleep on misleading/ambiguous context; defer hypothesis application when unreliable.

**v4.0 — Active inference:** select `observe`, `probe:<concept>`, or `sleep` by minimising expected free energy before committing to perception.

**v5.0 — Language grounding:** `TextGroundingBridge` + `EidosTextAgent` map phrases → 64-d vectors; `text_decision` surfaces cognitive outcomes.

**v5.1 — Hybrid spike:** `HybridEidosAgent` — LLM draft + EIDOS gate.

**v6.0 — Unified gate:** `GatePolicy` fuses cognitive steps, draft↔goal alignment, concept ambiguity.

**v6.1–v6.2 — Benchmark:** `AmbiguousQABenchmark` — 17 labeled cases (lab + real-world professional domains).

### 2.2 Component Map

| Component | Biological basis | Role |
|-----------|------------------|------|
| `PredictionEngine` | Predictive coding (Rao & Ballard 1999) | Nonlinear MLP; predict-compare-learn |
| `WorkspaceBuffer` | Global workspace (Baars 1988) | 7-slot active memory |
| `AssociationStore` | Hebbian learning (Hebb 1949) | Fast association graph |
| `AttentionGate` | Goal-directed attention | Salience routing |
| `RewardSignal` | Reward prediction error (Schultz 1997) | Intrinsic motivation |
| `SurpriseDetector` | Relative prediction error | System 2 trigger |
| `ReasoningLoop` | Dual-process theory (Kahneman 2011) | Hypothesis generation |
| `RecoveryContextTracker` | Episodic memory | Recent-context recovery (v1.4) |
| `EpisodicBuffer` | Hippocampus | Waking experience log (v2.0) |
| `BeliefGraph` | Neocortex | Slow consolidated beliefs (v2.0) |
| `SleepReplay` | Hippocampal replay (Wilson & McNaughton 1994) | Offline consolidation |
| `MetaCognitionMonitor` | Metacognition (Flavell; Nelson & Narens) | Context conflict + reasoning quality (v3) |
| `ActiveInferenceController` | Active inference (Friston et al. 2017) | EFE action selection: observe / probe / sleep (v4) |
| `TextGroundingBridge` | Symbol grounding (Harnad 1990); distributional semantics | Phrase → vector embedding (v5) |
| `EidosTextAgent` | Language adapter | `step_text`, `text_decision` (v5) |
| `GatePolicy` | Metacognitive control (Nelson & Narens 1990) | Unified hybrid gate (v6) |
| `HybridEidosAgent` | Dual-process (Kahneman 2011) | LLM + EIDOS sidecar (v5.1+) |
| `AmbiguousQABenchmark` | Calibration under uncertainty | Labeled defer/commit evaluation (v6.1+) |

### 2.3 Version Evolution

| Version | Key addition |
|---------|--------------|
| v1.0 | Full PAW implementation |
| v1.1 | Closed-loop reasoning; relative surprise; ablation flags |
| v1.2 | Nonlinear MLP; `consolidate_belief()` |
| v1.3 | `preview_consolidation()`; multi-concept scoring |
| v1.4 | Autonomous recovery probe inference |
| v2.0 | BeliefGraph + sleep replay |
| v3.0 | Meta-cognition: detect misleading context + ambiguous reasoning |
| v3.1 | Consequential meta-cognition: defer + auto-sleep |
| v4.0 | Active inference: expected free energy action selection |
| v5.0 | Text grounding bridge: phrases → PAW vectors |
| v5.1 | Hybrid LLM + EIDOS gate spike |
| v6.0 | Unified `GatePolicy` + optional SBERT embeddings |
| v6.1 | Ambiguous QA benchmark + Exp 22 end-to-end |
| v6.2 | Real-world benchmark expansion (10 professional domains) |

### 2.4 Ablation Flags

| Flag | Effect |
|------|--------|
| `enable_reasoning=False` | No ReasoningLoop |
| `apply_hypotheses=False` | Hypotheses not written back |
| `enable_meta_cognition=False` | v1.4/v2.0 recovery only (Exp 09–10) |
| `enable_meta_consequential=False` | v3.0 observe-only meta (flags, no defer/sleep) |
| `enable_active_inference=False` | No EFE action selection (default; Exp 01–14) |

---

## 3. Methods

- **Input space:** 64-dimensional vectors
- **No LLM APIs, no PyTorch/TensorFlow**
- **Seed:** 42 (reproducible)
- **Success experiments:** metric thresholds + ablation comparisons
- **Failure experiments (09–10):** pass when system fails predictably (meta off)

---

## 4. Results Summary

### Foundation (01–03)
Prediction learning, Hebbian associations, surprise-triggered reasoning — all pass.

### Reasoning consequentiality (04–06)
90.1% and 99.9% early-recovery improvement vs ablation.

### Disambiguation & autonomy (07–08)
3/3 multi-concept correct; autonomous inference without oracle.

### Failure modes (09–10)
Misleading context → wrong recovery (~5528 error). Cold start → no inference.

### CLS fix (11)
Sleep + BeliefGraph restores fire inference in both scenarios.

### Meta-cognition (12–13)
Exp 12: misleading context detected, fire recovered without sleep. Exp 13: ambiguous competition flagged.

### Consequential meta (14)
v3.1 defer/sleep beats v3.0 blind commit on ambiguous near-duplicate concepts.

### Active inference (15–16)
Exp 15: goal-directed epistemic probe selects `probe:fire` under ambiguity. Exp 16: v4 on probes and lowers error vs passive observe on cold ambiguous input.

### Text grounding (17–18)
Exp 17: `EidosTextAgent` + active inference probes goal-aligned text concept. Exp 18: pre-surprise sleep consolidates text session memory and fixes misleading phrase context.

### Hybrid + gate (19–22)
Exp 19: hybrid agent gates blind LLM commit. Exp 20: unified gate catches draft–goal misalignment. Exp 21: optional SBERT embeddings. Exp 22: full-stack (meta + sleep + gate) beats baseline; benchmark v2.0 covers lab + real-world domains.

---

## 5. Discussion

### 5.1 What worked

- Layered cognitive faculties compound: predict → reason → remember → doubt → *act on doubt*
- Every major claim has an ablation or failure-mode control
- Architecture remains interpretable (numpy-only, full state serialisation)

### 5.2 Limitations

| Limitation | Impact |
|------------|--------|
| 64-dim synthetic vectors | **Partially addressed in v5.0** — hash text embeddings; not full language |
| No action output | **Resolved in v4.0** — probe/sleep/observe via EFE |
| Sleep is explicit | No automatic circadian schedule |
| Single agent | No multi-agent workspace |
| Meta-cognition is heuristic | Not a learned uncertainty estimator |

### 5.3 Relationship to LLMs — Enhancement, Not Replacement

EIDOS does not compete with language models. It explores **orthogonal mechanisms**: structured deliberation under surprise, explicit memory systems, and calibrated self-monitoring.

**Implemented hybrid architecture (v5.1–v6.2):**

| Layer | Role |
|-------|------|
| **LLM** | Language interface, draft generation (`MockLanguageModel`, optional GPT-2) |
| **EIDOS module** | Surprise detection, episodic context, belief consolidation, meta-cognitive deferral, unified gate |

Integration patterns demonstrated in code:

1. **Cognitive wrapper** — LLM drafts scored by `GatePolicy`; misalignment triggers clarify/defer/probe.
2. **Memory sidecar** — `EpisodicBuffer` + `BeliefGraph` + `sleep()` consolidate session context (Exp 18, 22).
3. **Deferral gate** — Meta-cognition + benchmark labeled cases validate safe non-commit under ambiguity.

See `demos/hybrid_qa/` and `benchmark/ambiguous_qa/`.

---

## 6. Conclusion

EIDOS v6.2 adds a **real-world ambiguous QA benchmark** (12 professional domains) on top of the lab cases. The experiment remains a research prototype — not AGI, not a product — but a reproducible foundation for studying competence under uncertainty.

**Next research frontier:** learned gate thresholds from benchmark logs and belief-grounded LLM context.

---

## 7. Reproducibility

```bash
cd eidos
pip install -r requirements.txt
pytest tests/                    # 63+ unit tests
python run_all_experiments.py    # All 22 experiments + summary
py -m benchmark.ambiguous_qa.runner   # ambiguous QA benchmark report
```

State serialisation version: **6.0**

---

## Appendix A: Experiment Index

| # | Name | Type | Version |
|---|------|------|---------|
| 01 | Basic prediction | Success | v1.0 |
| 02 | Memory consolidation | Success | v1.0 |
| 03 | Reasoning chain | Success | v1.0 |
| 04 | Reasoning ablation | Success | v1.1 |
| 05 | Relational inference | Success | v1.1 |
| 06 | Reasoning recovery | Success | v1.2 |
| 07 | Multi-concept | Success | v1.3 |
| 08 | Autonomous recovery | Success | v1.4 |
| 09 | Misleading context | Failure mode | v1.4 |
| 10 | Cold start | Failure mode | v1.4 |
| 11 | CLS recovery | Success | v2.0 |
| 12 | Meta misleading context | Success | v3.0 A |
| 13 | Meta ambiguous reasoning | Success | v3.0 B |
| 14 | Meta consequential | Success | v3.1 |
| 15 | Active epistemic probe | Success | v4.0 |
| 16 | Active inference ablation | Success | v4.0 |
| 17 | Text goal-directed probe | Success | v5.0 |
| 18 | Text session memory | Success | v5.0 |
| 19 | Hybrid LLM gate | Success | v5.1 |
| 20 | Unified gate draft–goal | Success | v6.0 |
| 21 | SBERT embeddings | Success | v6.0 |
| 22 | End-to-end full stack | Success | v6.2 |

*Exp 01–18: core PAW lab. Exp 19–22: hybrid gate + benchmark.*

---

*Kisamapa Labs — Experiment 06 — EIDOS v6.2 — Lab Report*

# Kisamapa Labs — Experiment 06: EIDOS

**Emergent Intelligence via Distributed Organisational Systems**

| Field | Value |
|-------|-------|
| **Classification** | Research Prototype |
| **Status** | Complete (v3.0) |
| **Date** | June 2026 |
| **Stack** | Python, numpy, matplotlib, pytest |
| **Repository** | `eidos/` |

---

## Executive Summary

EIDOS is a laboratory prototype reasoning agent built from cognitive science primitives — not from transformer architectures or token prediction. Operating on 64-dimensional concept vectors, it implements a **Predictive Active Workspace (PAW)** architecture combining hierarchical predictive coding, global workspace broadcasting, Hebbian association learning, attentional gating, intrinsic curiosity reward, and a dual-process reasoning loop.

Across eleven controlled experiments, we demonstrate that:

1. **Prediction error decreases** with exposure to structured patterns (Exp 01).
2. **Associations form** from co-activation without supervision (Exp 02).
3. **System 2 deliberation activates** selectively under surprise (Exp 03–04).
4. **Reasoning is consequential** — it measurably improves recovery after model corruption (Exp 04, 06).
5. **Multi-concept disambiguation works** via consolidation preview scoring, without hardcoded routing (Exp 07).
6. **Recovery is end-to-end autonomous** — no external oracle hints required (Exp 08).
7. **Failure modes are documented and bounded** — misleading context and cold start break v1.4 recovery (Exp 09–10).
8. **Complementary learning systems fix those failures** — sleep replay into a BeliefGraph provides long-term recovery anchors (Exp 11).

**Conclusion:** Reasoning-like behaviour — surprise detection, hypothesis generation, belief revision, and measurable recovery — can emerge from a small set of biologically motivated mechanisms implemented in pure numpy, without an LLM.

---

## 1. Research Question

> Can an agent exhibit reasoning-like behaviour — maintaining context, learning associations, deliberating under surprise, and recovering from prediction failure — by composing cognitive science primitives, without statistical language modelling?

### Hypothesis

A minimal set of six interacting mechanisms (prediction, workspace, association, attention, reward, deliberation) is sufficient to produce measurable reasoning behaviour on vector-valued concept tasks, with ablation-controlled evidence that deliberation causally improves outcomes.

### Null hypothesis

Reasoning loop activation is decorative — removing it does not change recovery error or hypothesis quality.

**Result:** Rejected. Exp 04 and 06 show 90–99.9% recovery improvement with reasoning enabled vs ablated.

---

## 2. Architecture

### 2.1 Predictive Active Workspace (PAW)

```
INPUT → AttentionGate → WorkspaceBuffer → PredictionEngine → prediction_error
                                              ↓                    ↓
                                    AssociationStore ←──── ReasoningLoop (if surprised)
                                              ↓
                                    RewardSignal → OUTPUT
```

**v2.0 extension (Complementary Learning Systems):**

```
Waking steps → EpisodicBuffer ──sleep()──→ BeliefGraph (slow store)
                              └─ SleepReplay → AssociationStore + PredictionEngine
```

### 2.2 Component Map

| Component | Biological basis | Role |
|-----------|------------------|------|
| `PredictionEngine` | Predictive coding (Rao & Ballard 1999) | Nonlinear MLP; predict-compare-learn |
| `WorkspaceBuffer` | Global workspace (Baars 1988) | 7-slot limited active memory |
| `AssociationStore` | Hebbian learning (Hebb 1949) | Fast episodic association graph |
| `AttentionGate` | Goal-directed attention (Corbetta & Shulman 2002) | Salience routing into workspace |
| `RewardSignal` | Reward prediction error (Schultz 1997) | Intrinsic motivation via surprise reduction |
| `SurpriseDetector` | Relative prediction error | System 2 trigger (not fixed threshold) |
| `ReasoningLoop` | Dual-process theory (Kahneman 2011) | Hypothesis generation under surprise |
| `RecoveryContextTracker` | Episodic memory | Recent-context recovery targeting (v1.4) |
| `EpisodicBuffer` | Hippocampus | Waking experience log (v2.0) |
| `BeliefGraph` | Neocortex / semantic memory | Slow consolidated beliefs (v2.0) |
| `SleepReplay` | Hippocampal replay (Wilson & McNaughton 1994) | Offline consolidation (v2.0) |

### 2.3 Version Evolution

| Version | Key addition |
|---------|--------------|
| v1.0 | Full PAW implementation |
| v1.1 | Closed-loop reasoning; relative surprise; ablation flags |
| v1.2 | Nonlinear MLP; `consolidate_belief()`; Exp 06 recovery proof |
| v1.3 | `preview_consolidation()`; multi-concept scoring; no hardcoded overrides |
| v1.4 | Autonomous recovery probe inference from recent context |
| v2.0 | BeliefGraph + sleep replay; merged slow/fast recovery inference |

---

## 3. Methods

### 3.1 Environment

- **Input space:** 64-dimensional real vectors (`INPUT_DIM = 64`)
- **Concepts:** Registered label → prototype vector mappings
- **No external APIs, no PyTorch/TensorFlow, no language tokens**
- **Random seed:** 42 (reproducible across experiments)

### 3.2 Evaluation Protocol

| Experiment type | Pass criterion |
|-----------------|----------------|
| Success (01–08, 11) | Metric threshold met (error reduction, correct hypothesis, beats ablation) |
| Failure mode (09–10) | System fails predictably (wrong inference or no recovery benefit) |

### 3.3 Ablation Design

Two flags control reasoning behaviour:
- `enable_reasoning=False` — ReasoningLoop never runs
- `apply_hypotheses=False` — hypotheses generated but not written back to workspace/weights

Identical agents with different flags enable causal comparison.

---

## 4. Results

### 4.1 Foundation (Exp 01–03)

| Exp | Question | Result |
|-----|----------|--------|
| **01** Basic prediction | Does error decrease on structured patterns? | ~48–21% reduction over 200 steps |
| **02** Memory consolidation | Does Hebbian learning capture co-occurrence? | `cat → dog` becomes top associate |
| **03** Reasoning chain | Does System 2 activate on surprise? | ReasoningLoop fires; hypothesis returned |

### 4.2 Reasoning Consequentiality (Exp 04–06)

| Exp | Metric | Full agent | Ablated | Improvement |
|-----|--------|------------|---------|-------------|
| **04** Early recovery error | Mean steps 1–5 post-surprise | 0.42 | 4.24 | **90.1%** |
| **06** Early recovery error | After weight corruption | 0.58 | 563.83 | **99.9%** |

Exp 06 confirms that reasoning + belief consolidation repairs deliberately corrupted prediction weights — the effect is not cosmetic.

### 4.3 Disambiguation & Autonomy (Exp 07–08)

| Exp | Scenario | Outcome |
|-----|----------|---------|
| **07** fire / water / smoke | 3/3 correct concept selected via consolidation preview | PASS |
| **08** Same as 07, zero external `recovery_probe` | 3/3 inferred from `recent_history` | PASS |

### 4.4 Documented Failure Modes (Exp 09–10)

These experiments **pass when the system fails** — establishing bounded limits of v1.4 autonomous recovery.

| Exp | Failure condition | Observed behaviour |
|-----|-------------------|-------------------|
| **09** Misleading context | 15 decoy warmup steps (water) after training on fire | Infers `water`, error ~5528 vs baseline ~27 |
| **10** Cold start | Cleared workspace + history, zero warmup | `inference_source=none`, no recovery benefit |

**Interpretation:** v1.4 recovery depends entirely on recent episodic context quality. It is not magic.

### 4.5 CLS Fix (Exp 11)

| Scenario | v1.4 (Exp 09/10) | v2.0 (Exp 11) |
|----------|------------------|---------------|
| Misleading context | Wrong inference, error ~5528 | `belief_graph` → fire, error **0.31** |
| Cold start | No inference, error ~3436 | `belief_graph` → fire, error **0.22** |

Sleep replay after training consolidates long-term beliefs that anchor recovery when short-term context is missing or misleading.

---

## 5. Discussion

### 5.1 What worked

- **Relative surprise** prevents ReasoningLoop from firing every step — selective System 2 engagement.
- **Consolidation preview** enables principled multi-concept disambiguation without hardcoded routing.
- **Belief consolidation** makes reasoning consequential — hypotheses change prediction weights, not just logs.
- **Complementary learning systems** provide the missing long-term memory layer that v1.4 lacked.

### 5.2 Limitations

| Limitation | Impact |
|------------|--------|
| 64-dim synthetic vectors | No language, vision, or real-world grounding |
| No action output | Agent predicts and reasons but does not choose external actions |
| Sleep must be called explicitly | No automatic consolidation schedule |
| Single-agent only | No multi-agent communication or shared workspace |
| numpy MLP, not spiking | Biologically inspired, not biologically accurate |
| Weight corruption stress test | Cold start with corruption still challenging; Exp 11 uses lr=0 on surprise for cold scenario |

### 5.3 Relationship to LLMs

EIDOS does not compete with language models. It explores an orthogonal hypothesis: that structured deliberation under surprise can emerge from predictive processing and association, without next-token prediction. The transparency of every component is the point.

---

## 6. Conclusion

EIDOS v2.0 demonstrates that a small, interpretable cognitive architecture can:

- Learn predictive models online
- Form associations from co-activation
- Deliberate selectively under surprise
- Recover from prediction failure measurably better than ablated controls
- Disambiguate among multiple concepts without oracle hints
- Fail predictably when memory conditions are adverse
- Recover from those failures via complementary learning systems

The experiment is **complete** as a laboratory prototype. Future expansion paths (meta-cognition, active inference actions, language grounding) are documented but out of scope for this release.

---

## 7. Reproducibility

```bash
cd eidos
pip install -r requirements.txt
pytest tests/                    # 38 unit tests
python run_all_experiments.py    # All 11 experiments + summary
```

Individual experiments: see `README.md` § Running Experiments.

State serialisation version: `2.0` (includes BeliefGraph, EpisodicBuffer, RecoveryContext).

---

## 8. References

Primitives extracted from 15 source papers. Full bibliography in `research/knowledge_base.json` and `research/primitives/`.

Key frameworks: Friston (2010, 2017), Rao & Ballard (1999), Baars (1988), Baddeley (2003), Hebb (1949), McClelland et al. (1995), Wilson & McNaughton (1994), Kahneman (2011), Gentner (1983), Schultz (1997), Sutton & Barto (2018).

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

---

## v3.0 Addendum — Meta-Cognition

### A: Misleading context detection
Compares recent window dominance against full `EpisodicBuffer` distribution. When short-term context is a decoy burst (Exp 09 scenario), meta-cognition overrides with `meta_cognition` inference source and flags `misleading_context_detected`.

### B: Reasoning trace monitoring
After ReasoningLoop selection, evaluates confidence and top-2 preview error gap. Flags `ambiguous_hypothesis` when competition is too close; may suppress hypothesis application when both ambiguous and low-confidence.

### Evidence
- **Exp 12:** Misled warmup (15 water, trained fire) → infers fire, error < 50, flag raised
- **Exp 13:** Near-duplicate concepts → `ambiguous_hypothesis` flagged

Exp 09–10 remain valid with `enable_meta_cognition=False`.

---

*Kisamapa Labs — Experiment 06 — EIDOS v3.0 — Lab Report*

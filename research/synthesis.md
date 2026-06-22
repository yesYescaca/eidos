# EIDOS Research Synthesis

*Phase 1 output — Kisamapa Labs Experiment 06*

---

## 1. What is the minimal mechanism set for reasoning-like behaviour?

Reasoning-like behaviour — the ability to maintain context, predict outcomes, learn from experience, and deliberate when surprised — requires at minimum **six interacting mechanisms**:

| # | Mechanism | Source Primitive | Role |
|---|-----------|------------------|------|
| 1 | **Hierarchical prediction** | Rao & Ballard (1999), Friston (2010) | Generate expectations; detect surprise |
| 2 | **Limited global workspace** | Baars (1988), Baddeley (2003) | Hold active context for all modules |
| 3 | **Associative learning** | Hebb (1949) | Build relational knowledge from co-activation |
| 4 | **Attentional gating** | Corbetta & Shulman (2002) | Select what enters workspace |
| 5 | **Intrinsic reward** | Schultz (1997), Sutton & Barto (2018) | Motivate surprise reduction |
| 6 | **Deliberate reasoning loop** | Kahneman (2011), Gentner (1983) | Handle high-surprise cases via hypothesis testing |

Without prediction, there is no surprise signal. Without workspace, there is no shared context. Without associations, deliberation has no material to work with. Without attention, the workspace floods. Without reward, there is no learning pressure. Without the reasoning loop, the system cannot recover from prediction failure.

**Secondary but valuable mechanisms** (not in v1 minimal set):
- Complementary learning systems (McClelland 1995) — slow consolidation
- Hippocampal replay (Wilson & McNaughton 1994) — offline memory strengthening
- Active inference action selection (Friston 2017) — acting to confirm predictions
- Biased competition (Desimone & Duncan 1995) — finer-grained suppression

---

## 2. Which mechanisms are most directly implementable?

**Tier 1 — Direct code mapping (implemented in v1):**

1. **Hebbian association graph** — trivial `defaultdict` with pair-wise weight updates and decay. Fully deterministic, serialisable.
2. **Fixed-capacity workspace buffer** — list with eviction policy. No ML required.
3. **Two-layer linear predictive coding** — numpy matrix operations + SGD. Transparent and testable.
4. **Salience-weighted attention gate** — cosine similarity/distance on vectors. Pure linear algebra.
5. **Intrinsic reward from error reduction** — scalar subtraction on rolling window.
6. **Threshold-triggered reasoning loop** — conditional branch with hypothesis enumeration over association graph.

**Tier 2 — Implementable with moderate effort (v2.0 implemented):**

7. **Complementary learning systems** — `EpisodicBuffer` + `BeliefGraph` + `SleepReplay` (v2.0)
8. **TD value learning** — extend RewardSignal with state-value table
9. **Structure-mapping analogy** — graph isomorphism over association subgraphs

**Tier 3 — Hard to implement faithfully:**

10. **Full active inference with expected free energy** — requires generative model over actions and states
11. **Neuronal ignition dynamics** (Dehaene 1998) — needs recurrent nonlinear dynamics
12. **True System 2 serial processing** — biological System 2 is slow and capacity-limited in ways hard to capture without real time costs

---

## 3. Proposed architecture for EIDOS

EIDOS implements the **Predictive Active Workspace (PAW)** architecture:

```
INPUT → AttentionGate → WorkspaceBuffer → PredictionEngine → prediction_error
                                    ↓                              ↓
                            AssociationStore ← RewardSignal ← (error reduction)
                                    ↓
                            ReasoningLoop (if error > threshold)
                                    ↓
                              OUTPUT / ACTION
```

**Design rationale:**

- **PredictionEngine is the spine.** Friston's free-energy principle and Rao-Ballard predictive coding provide the core loop: predict, compare, learn. Everything else modulates this loop.
- **WorkspaceBuffer is the stage.** Baars' global workspace makes active contents globally readable. All components read from `broadcast()`.
- **AttentionGate is the door.** Corbetta-Shulman dual attention selects what deserves workspace access based on surprise and goal relevance.
- **AssociationStore is fast memory.** Hebbian co-activation builds the relational structure that ReasoningLoop queries.
- **RewardSignal provides intrinsic drive.** Schultz RPE adapted: reward = surprise reduction. The agent is curious — it wants to predict better.
- **ReasoningLoop is slow thinking.** Kahneman's System 2 activates only when System 1 (prediction) fails. It generates hypotheses from associations and selects the one minimising predicted error.

**Why PAW over alternatives:**
- Maps cleanly to numpy implementations without ML frameworks
- Each component is independently testable
- Biological grounding is explicit and traceable to source papers
- Scales from 64-dim toy vectors to richer representations later

---

## 4. Key risks and unknowns

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Linear models may not capture nonlinear structure** | High | Sufficient for prototype; hierarchical depth can increase later |
| **ReasoningLoop hypothesis generation is simplistic** | Medium | Uses association graph paths; adequate for surprise response demo |
| **No true language grounding** | High | Concepts are string labels + vectors; no semantic composition yet |
| **Workspace capacity is arbitrary** | Low | Miller's 7 is configurable; 4 may be more realistic per Cowan |
| **Hebbian store lacks CLS slow consolidation** | Medium | **Resolved in v2.0** — BeliefGraph + sleep replay |
| **Intrinsic reward may not drive useful exploration** | Medium | Combined with extrinsic goal reward in agent loop |
| **Catastrophic forgetting in prediction weights** | Medium | Online SGD on non-stationary input; association decay helps graph side |
| **System 1/2 threshold is a hyperparameter** | Low | Threshold=0.5 is tunable; experiments validate sensitivity |
| **No action output in v1** | Medium | Active inference action selection deferred to future extension |
| **Biological plausibility ≠ cognitive adequacy** | High | This is a laboratory model, not a brain simulation |

**Fundamental unknown:** Whether these mechanisms jointly produce *emergent* reasoning or merely simulate its surface signatures on toy tasks. The experiments test the minimum viable claim: learning, association, and deliberation under surprise.

---

## v1.1 Update (Closed-Loop Reasoning)

v1.1 adds:

1. **SurpriseDetector** — relative surprise triggering (error spike vs rolling baseline)
2. **Hypothesis feedback** — selected hypotheses write to workspace, boost associations, bias prediction context
3. **Structure-based hypotheses** — association paths and `explain:concept` toward registered vectors
4. **Ablation flags** — `enable_reasoning` / `apply_hypotheses` for controlled comparisons
5. **Exp 04** — selective reasoning activation + correct hypothesis under ablation
6. **Exp 05** — relational path inference (`cat->dog->bone`)

**Open question for v1.2:** Recovery improvement after surprise remains marginal with linear predictors. Next step: stronger belief integration or nonlinear generative models.

---

## v1.2 Update (Belief Consolidation)

v1.2 adds:

1. **Nonlinear MLP PredictionEngine** — tanh hidden units, numpy-only backprop
2. **`consolidate_belief()`** — offline weight replay teaching the model that a known concept predicts itself
3. **Anomaly anchoring** — on high-surprise unregistered inputs, override to best registered concept when available
4. **Exp 06** — after deliberate weight corruption, full agent recovers ~99.9% better than ablation (PASS)

**Evidence of consequential reasoning:** Exp 06 demonstrates that reasoning + consolidation produces measurably lower prediction error than identical agent without reasoning, on the same corrupted weights.

---

## v1.3 Update (Consolidation Preview — No Hardcoded Overrides)

v1.3 removes the v1.2 `anomaly → fire` override and replaces it with:

1. **`preview_consolidation()`** — simulate belief replay without modifying live weights
2. **All registered concepts scored** — ReasoningLoop picks the concept with lowest post-consolidation probe error
3. **Exp 07** — three concepts (fire/water/smoke), each trained separately; reasoning must pick the correct one

**Evidence upgrade:** Reasoning now earns the correct answer through consolidation preview, not hardcoded routing.

---

## v1.4 Update (Autonomous Recovery Targeting)

v1.4 removes the last external hint (`recovery_probe` passed from experiments):

1. **`RecoveryContextTracker`** — maintains rolling episodic history of (label, vector) pairs
2. **`_resolve_recovery_probe()`** — infers recovery target from dominant recent registered concept; falls back to workspace vote or goal similarity
3. **Exp 08** — identical multi-concept scenario to Exp 07 but with zero external probes; agent must infer target from warmup + training context

**Evidence upgrade:** The full recovery loop is now end-to-end autonomous — surprise detection, target inference, consolidation preview scoring, and belief application require no oracle hints.

---

## Failure-Mode Experiments (v1.4)

Exp 09 and Exp 10 document **when autonomous recovery breaks**:

| Experiment | Failure condition | Observed behaviour |
|------------|-------------------|-------------------|
| **Exp 09** | Recent history dominated by decoy concept | Infers wrong label, selects wrong associate, recovery error >> baseline |
| **Exp 10** | Cleared workspace + history, zero warmup | `inference_source=none`, no recovery benefit vs ablated agent |

**Why this matters:** Success experiments (04–08) prove the mechanism works; failure experiments prove it is not magic — episodic context quality is a hard dependency.

---

## v2.0 Update (Complementary Learning Systems)

v2.0 implements the slow memory system that failure-mode analysis demanded:

1. **`EpisodicBuffer`** — logs waking experience for offline replay
2. **`BeliefGraph`** — slow semantic store (concept strength + prototypes)
3. **`SleepReplay`** + **`agent.sleep()`** — hippocampal→cortical consolidation pass
4. **Merged recovery inference** — BeliefGraph overrides misleading recent context; fills gap when recent context is empty
5. **Exp 11** — misleading context and cold start both recover correctly after sleep

**Evidence upgrade:** The agent no longer depends solely on "what happened in the last 20 steps." Long-term consolidated beliefs anchor recovery when episodic context fails.

---

## Project Status: Active (v6.2)

EIDOS Experiment 06 continues as an active research prototype. Full writeup: `LAB_REPORT.md`.

| Deliverable | Status |
|-------------|--------|
| 15 cognitive primitives | Done |
| PAW architecture (13+ components) | Done |
| 22 experiments (success + failure + hybrid) | Done |
| Meta-cognition (v3.0) | Done |
| Consequential meta (v3.1) | Done |
| Active inference (v4.0) | Done |
| Text grounding (v5.0) | Done |
| Hybrid LLM gate (v5.1–v6.0) | Done |
| Ambiguous QA benchmark (v6.1–v6.2) | Done |
| 63 unit tests | Done |
| Lab report | Done |

---

## v4.0 Update (Active Inference)

1. **`ActiveInferenceController`** — expected free energy over observe / probe / sleep
2. **Exp 15–16** — epistemic probing and ablation vs passive observe

---

## v5.0 Update (Language Grounding)

1. **`TextGroundingBridge`** — hash n-gram embeddings (numpy only)
2. **`EidosTextAgent`** — `step_text`, `text_decision`
3. **Exp 17–18** — goal-directed text probe + text session memory

---

## v6.2 Real-World Benchmark

1. **`cases.json` v2.0** — 17 cases: lab + IT, support, security, finance, clinical, HR, legal, aviation, education, logistics
2. **Per-category metrics** — `by_category`, `by_domain`, filter API
3. **Exp 22 updated** — reports real-world subset safety alongside full benchmark

---

## Future: LLM Enhancement (Research Direction)

EIDOS is orthogonal to LLMs: predictive memory, deliberation under surprise, and meta-cognitive deferral complement language fluency. The hybrid spike (v5.1), unified gate (v6.0), and benchmark (v6.1–v6.2) demonstrate this — see `demos/hybrid_qa/`.

*Synthesis updated for v6.2 real-world benchmark.*

# EIDOS

**Emergent Intelligence via Distributed Organisational Systems**

**Status: Complete (v2.0)** — See [LAB_REPORT.md](LAB_REPORT.md) for full findings.

EIDOS is a laboratory prototype reasoning agent built from cognitive science primitives — not from transformer architectures or token prediction. Instead of learning statistical text patterns, EIDOS implements mechanisms drawn from neuroscience: hierarchical predictive coding, global workspace broadcasting, Hebbian association learning, attentional gating, and intrinsic curiosity reward. It is a transparent, numpy-only system designed to explore how biological cognition might be computationally reconstructed.

## Biological Frameworks

| Framework | Role in EIDOS |
|-----------|---------------|
| **Predictive Processing** (Friston, Rao & Ballard) | Core predict-compare-learn loop |
| **Global Workspace Theory** (Baars, Dehaene) | Limited-capacity active memory broadcast |
| **Complementary Learning Systems** (McClelland) | Fast episodic + slow semantic (`BeliefGraph`) |
| **Hebbian Learning** (Hebb) | Association graph from co-activation |
| **Dual-Process Theory** (Kahneman) | Fast prediction + slow deliberation |
| **Reward Prediction Error** (Schultz) | Intrinsic motivation via surprise reduction |

## Folder Structure

```
eidos/
├── LAB_REPORT.md      # Full experiment writeup (Kisamapa Labs)
├── research/          # Cognitive primitives (JSON) + synthesis
├── architecture/      # Component implementations
├── agent/             # Main EidosAgent class
├── tests/             # pytest suite (38 tests)
├── experiments/       # Eleven validation experiments (v1 → v2.0)
└── run_all_experiments.py
```

## v2.0 — Complementary Learning Systems (BeliefGraph + Sleep Replay)

Version 2.0 adds the slow memory system from McClelland's CLS theory:

- **`EpisodicBuffer`** — hippocampal log of waking (label, vector) traces
- **`BeliefGraph`** — slow semantic store with concept strength + prototype vectors
- **`SleepReplay`** — offline pass replaying episodic traces into BeliefGraph
- **`agent.sleep()`** — run consolidation after training episodes
- **Merged recovery inference** — when recent context conflicts with slow store, BeliefGraph wins if strength ratio exceeds threshold; when recent context is empty, BeliefGraph provides the probe
- **Exp 11** — demonstrates v2.0 fixes both Exp 09 (misleading context) and Exp 10 (cold start) failure modes

## v1.4 — Autonomous Recovery Targeting

Version 1.4 removes the last external cheat: experiments no longer pass `recovery_probe` on surprise steps.

- **`RecoveryContextTracker`** — rolling window of recent (label, vector) steps
- **`_resolve_recovery_probe()`** — infers recovery target from dominant recent registered concept, workspace fallback, or goal similarity
- **Exp 08** — same multi-concept test as Exp 07 but with zero external hints; must infer correct concept from episodic context

## Failure-Mode Experiments (v1.4)

These experiments **pass when the system fails predictably** — documenting limits of autonomous recovery:

- **Exp 09** — misleading recent context (decoy warmup steps) causes wrong inference and degraded recovery
- **Exp 10** — cold start (cleared workspace + history, zero warmup) yields no inference and no recovery benefit vs ablation

## v1.3 — Consolidation Preview (No Hardcoded Overrides)

Version 1.3 removes the v1.2 anomaly→fire cheat and replaces it with **consolidation preview scoring**:

- **`preview_consolidation()`** — simulates weight replay on a snapshot, returns probe error without modifying live weights
- **ReasoningLoop** scores every registered concept via preview; picks lowest recovery error
- **Exp 07** — fire / water / smoke disambiguation: must pick the trained concept without hardcoding

## v1.2 — Belief Consolidation + Nonlinear Prediction

Version 1.2 makes reasoning **measurably improve outcomes**:

- **Nonlinear MLP predictor** — tanh hidden layers (numpy-only, no frameworks)
- **Belief consolidation** — `consolidate_belief()` replays the chosen concept into prediction weights (hippocampal→cortical transfer)
- **Exp 06** — reasoning-on must beat ablation by ≥10% after weight corruption

## v1.1 — Closed-Loop Reasoning

Version 1.1 makes reasoning **consequential**:

- **Relative surprise** (`SurpriseDetector`) — System 2 triggers only on error spikes vs rolling baseline, not every step
- **Hypothesis feedback** — selected hypotheses are written to workspace, strengthen associations, and bias the next prediction
- **Structure-based hypotheses** — generated from association graph paths, not random noise
- **Ablation flags** — `enable_reasoning=False` / `apply_hypotheses=False` for controlled comparisons

## Quick Start

```bash
cd eidos
pip install -r requirements.txt
pytest tests/                      # 38 unit tests
python run_all_experiments.py      # All 11 experiments + summary
```

## Running Experiments (individual)

```bash
python experiments/exp_01_basic_prediction/run.py
python experiments/exp_02_memory_consolidation/run.py
python experiments/exp_03_reasoning_chain/run.py
python experiments/exp_04_reasoning_ablation/run.py   # v1.1: reasoning ON vs OFF
python experiments/exp_05_relational_inference/run.py  # v1.1: cat->dog->bone chain
python experiments/exp_06_reasoning_recovery/run.py  # v1.2: reasoning must beat ablation
python experiments/exp_07_multi_concept/run.py       # v1.3: multi-concept disambiguation
python experiments/exp_08_autonomous_recovery/run.py # v1.4: no external recovery hints
python experiments/exp_09_misleading_context/run.py  # failure: decoy episodic context
python experiments/exp_10_cold_start/run.py            # failure: no context / cold start
python experiments/exp_11_cls_recovery/run.py          # v2.0: sleep replay fixes 09+10
```

## Running Tests

```bash
pytest tests/
```

## Running the Agent

```bash
python agent/run.py
```

## Cognitive Primitives

15 primitives extracted from 15 source papers across 6 categories:

**Prediction:** Free Energy Minimisation (Friston 2010), Active Inference (Friston 2017), Hierarchical Predictive Coding (Rao & Ballard 1999)

**Memory:** Multicomponent Working Memory (Baddeley 2003), Global Workspace Broadcasting (Baars 1988), Neuronal Global Workspace (Dehaene 1998)

**Learning:** Hebbian Association (Hebb 1949), Complementary Learning Systems (McClelland 1995), Hippocampal Replay (Wilson & McNaughton 1994)

**Attention:** Goal-Directed Attention (Corbetta & Shulman 2002), Biased Competition (Desimone & Duncan 1995)

**Reasoning:** Dual-Process Theory (Kahneman 2011), Structure-Mapping Analogy (Gentner 1983)

**Reward:** Reward Prediction Error (Schultz 1997), Temporal Difference Learning (Sutton & Barto 2018)

Full entries in `research/primitives/` and index in `research/knowledge_base.json`.

## What This Is Not

- **Not AGI.** EIDOS is a minimal laboratory model operating on 64-dimensional vectors.
- **Not an LLM replacement.** It does not process language or predict tokens.
- **Not biologically accurate.** It is inspired by neuroscience, not simulating neurons.
- **Not production-ready.** No API, no deployment, no scaling claims.

EIDOS is a tool for understanding cognition computationally — a Kisamapa Labs experiment in building intelligence from first principles.

## Future Expansion (out of scope for v2.0)

Potential directions when revisiting this project: meta-cognition (reasoning trace monitoring), active inference action selection, language-to-vector grounding, multi-agent shared workspace. See `SPEC.md` §12 and `LAB_REPORT.md` §5.2.

---

*KISAMAPA LABS — EXPERIMENT 06 — EIDOS v2.0*
*Classification: Research Prototype — Complete*

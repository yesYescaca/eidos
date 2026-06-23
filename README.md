# EIDOS

**Emergent Intelligence via Distributed Organisational Systems**

**Status: Active (v7.6)** — [Lab report](LAB_REPORT.md) · [Changelog](CHANGELOG.md) · [TruthfulQA eval](docs/TRUTHFULQA_EVAL.md) · [Live pilot](docs/LIVE_EVAL_PILOT.md) · [Multi-model eval](docs/MULTIMODEL_EVAL.md) · [Releases](https://github.com/yesYescaca/eidos/releases)

EIDOS is a laboratory prototype reasoning agent built from cognitive science primitives — not from transformer architectures or token prediction. Instead of learning statistical text patterns, EIDOS implements mechanisms drawn from neuroscience: hierarchical predictive coding, global workspace broadcasting, Hebbian association learning, attentional gating, and intrinsic curiosity reward. It is a transparent, numpy-only system designed to explore how biological cognition might be computationally reconstructed.

## At a glance

| | |
|---|---|
| **Experiments** | 28 controlled tests (v1 → v7.6) |
| **Unit tests** | 105+ (pytest) |
| **Benchmark** | 17 ambiguous QA + EIDOS-Eval + TruthfulQA + mixed N=50 |
| **Latest** | v7.6 — multi-model live eval |
| **Hybrid** | LLM proposes → EIDOS gates (`HybridEidosAgent`) |

```bash
git clone https://github.com/yesYescaca/eidos.git && cd eidos
pip install -r requirements.txt
pytest tests/ && python run_all_experiments.py
py -m benchmark.ambiguous_qa.runner
py -m eval.eidos_eval.runner
py -m eval.eidos_eval.live_runner --provider groq --truthfulqa
py -m eval.eidos_eval.live_runner --provider groq --mixed
py -m eval.eidos_eval.run_multimodel_eval --provider groq
```

## v7.6 — Multi-Model Live Eval

- **`--model`** on live runner — per-model cache + reports
- **`run_multimodel_eval`** — TruthfulQA + mixed across 70B Llama / 8B Llama / GPT-OSS-20B
- See [docs/MULTIMODEL_EVAL.md](docs/MULTIMODEL_EVAL.md)

```bash
py -m eval.eidos_eval.live_runner --provider groq --model llama-3.1-8b-instant --truthfulqa
py -m eval.eidos_eval.run_multimodel_eval --provider groq --benchmarks truthfulqa mixed
```

## v7.5 — Mixed Benchmark + Abstention Calibration

- **`LIVE_TRUTHFULQA_V75`** — only abstain on genuine underdetermination
- **`questions_mixed_50.json`** — 25 misconceptions + 25 ambiguous
- **Headline metric** — `misconception_commit_ti_rate` (belief vs CoT on commits)

```bash
py -m eval.eidos_eval.build_mixed_eval_50
py -m eval.eidos_eval.live_runner --provider groq --mixed --limit 10
py -m eval.eidos_eval.live_runner --provider groq --truthfulqa
```

## v7.4 — TruthfulQA-Aligned Grading + Factual Gate

- **`truthfulqa_scorer.py`** — T / I / TI metrics (Lin et al., ACL 2022)
- **`LIVE_TRUTHFULQA`** gate profile — fewer spurious abstentions on misconceptions
- **N=50 Groq pilot** — belief TI 78% vs CoT 64% ([docs/LIVE_EVAL_PILOT.md](docs/LIVE_EVAL_PILOT.md))
- **Exp 26** — TruthfulQA grading mock CI

```bash
py -m eval.eidos_eval.live_runner --provider groq --truthfulqa --limit 10
py -m eval.eidos_eval.live_runner --provider groq --truthfulqa
```

## v7.3 — TruthfulQA N=50 + CoT Baseline

- **`questions_truthfulqa_50.json`** — 50 Misconceptions items from official TruthfulQA CSV
- **`llm_cot` mode** — chain-of-thought baseline vs EIDOS belief
- **`gate_profiles.py`** — calibrated gate-only thresholds (less over-abstention)
- **Pilot results** — [docs/LIVE_EVAL_PILOT.md](docs/LIVE_EVAL_PILOT.md)
- **Exp 25** — TruthfulQA meta vs CoT (CI subset)

```bash
py -m eval.eidos_eval.build_truthfulqa_subset --n 50
py -m eval.eidos_eval.live_runner --provider groq --truthfulqa
py -m eval.eidos_eval.calibrate_gate
```

## v7.2 — Calibrated Live Sidecar

- **task_accuracy** metric, SBERT live path, response cache

```bash
py -m eval.eidos_eval.live_runner --provider groq
```

## v7.1 — Live Groq Eval + Belief-Grounded Sidecar

- **`create_live_llm("groq")`** — live API via Groq (`GROQ_API_KEY`)
- **`enable_belief_context`** — inject EIDOS concept rankings into LLM prompt
- **Draft–concept mismatch** — gate vetoes drafts aligned to wrong concept
- **Exp 24** — optional live Groq comparison (skips in CI without API key)

```bash
set GROQ_API_KEY=gsk_...
py -m eval.eidos_eval.live_runner --provider groq
py experiments/exp_24_groq_live_eval/run.py
py demos/hybrid_qa/run.py --groq --meta-injection --belief-context
```

## v7.0 — EIDOS-Eval + Metacognitive Injection

- **`eval/eidos_eval/`** — LLM-alone vs EIDOS gate vs meta-injection comparison
- **`enable_meta_injection`** — feed cognitive monitor signal back into LLM revision
- **`OpenAICompatibleLLM`** — optional API eval (`OPENAI_API_KEY`)
- **Exp 23** — gate reduces false commits vs LLM-alone on graded questions

See [docs/POSITIONING.md](docs/POSITIONING.md) for Core vs Sidecar.

```bash
py experiments/exp_23_eidos_eval/run.py
```

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
├── architecture/      # PAW components + gate + bridge + hybrid
├── agent/             # EidosAgent, EidosTextAgent
├── benchmark/         # Ambiguous QA benchmark (v6.1–v6.2)
├── demos/             # Hybrid QA demo (LLM + EIDOS gate)
├── tests/             # pytest suite (70 tests)
├── docs/              # WHAT_EIDOS_IS.md, POSITIONING.md, version plans
├── eval/              # EIDOS-Eval harness (v7.0)
├── experiments/       # Twenty-three validation experiments (v1 → v7.0)
└── run_all_experiments.py
```

## v6.2 — Real-World Benchmark

Expands the ambiguous QA benchmark with **12 real-world professional scenarios**:

| Domain | Example ambiguity |
|--------|-------------------|
| IT support | Password reset vs account compromise |
| Cybersecurity | Phishing vs marketing email |
| Finance | Fraud vs duplicate billing |
| Clinical | Stroke vs migraine triage |
| HR / legal / aviation / education / logistics | …see [benchmark README](benchmark/ambiguous_qa/README.md) |

- **`cases.json` v2.0** — 17 total cases (5 lab + 12 real-world)
- **Per-category metrics** — `by_category`, `by_domain`, filter API
- **Exp 22** — reports full benchmark + real-world subset safety

```bash
py -m benchmark.ambiguous_qa.runner
```

## v6.1 — Ambiguous QA Benchmark + Exp 22

Reusable labeled benchmark for hybrid gate evaluation:

- **`AmbiguousQABenchmark`** — decision match, false-commit, per-case audit
- **Exp 22** — end-to-end: misleading context → sleep → meta + active inference + unified gate

```bash
py experiments/exp_22_end_to_end/run.py
```

## v6.0 — Unified Gate

Single `GatePolicy` fuses meta-cognition, active inference, and text alignment:

- **`GatePolicy`** — draft↔goal alignment + concept ambiguity + cognitive merge
- **`create_grounding("hash" | "sbert")`** — optional sentence-transformers backend
- **Exp 20–21** — unified gate vs legacy merge; SBERT separation

```bash
py experiments/exp_20_unified_gate/run.py
pip install -r requirements-embeddings.txt  # optional SBERT
py experiments/exp_21_sbert_embeddings/run.py
```

## v5.1 / Hybrid Spike — LLM + EIDOS

Optional layer (not required for core experiments):

- **`HybridEidosAgent`** — LLM draft → EIDOS monitor → gate (`commit` / `defer` / `clarify` / `probe`)
- **`MockLanguageModel`** — CI-safe; **`GPT2LanguageModel`** — CPU demo
- **Exp 19** — gate blocks blind LLM commit on ambiguous input
- **Exp 20** — unified gate catches draft–goal misalignment (v6.0)
- **Exp 21** — optional SBERT embeddings beat hash separation (v6.0)
- **Demo:** `demos/hybrid_qa/run.py`

## v5.0 — Language Grounding Bridge

Version 5.0 connects natural language to PAW **without** an LLM or GPU:

- **`TextGroundingBridge`** — hashed n-gram embeddings → 64-d vectors (numpy only)
- **`EidosTextAgent`** — `register_text_concept()`, `step_text()`, `text_decision`
- **Decisions:** `observe` | `probe` | `defer` | `clarify` | `commit` | `sleep`
- **Exp 17** — goal-directed text probing via active inference
- **Exp 18** — text session memory: sleep fixes misleading phrase context

## v4.0 — Active Inference

Version 4.0 closes the perception–action loop (Friston et al. 2017):

- **`ActiveInferenceController`** — minimises expected free energy over discrete actions
- **Actions:** `observe` (passive), `probe:<concept>` (epistemic sampling), `sleep` (offline consolidation)
- **`enable_active_inference`** — default off for Exp 01–14 compatibility; on in Exp 15–16
- **Exp 15** — goal-directed epistemic probe under ambiguity
- **Exp 16** — v4 on beats v4 off on cold ambiguous input

## v3.1 — Consequential Meta-Cognition

Version 3.1 makes meta-cognition **act**, not only observe:

- **`enable_meta_consequential`** — when off, v3.0 observe-only behaviour (Exp 14 ablation)
- **Defer hypothesis** on `ambiguous_hypothesis` or `low_confidence` alone
- **Auto `sleep()`** on misleading context and after ambiguous reasoning (retry)
- **Exp 14** — v3.1 defer/sleep beats v3.0 blind commit on ambiguous recovery

## v3.0 — Meta-Cognition

Version 3.0 adds System 2 self-monitoring:

- **`MetaCognitionMonitor`** — detects misleading short-term context vs long episodic evidence (A)
- **Reasoning quality flags** — `ambiguous_hypothesis`, `low_confidence`, `hypothesis_suppressed` (B)
- **`enable_meta_cognition`** — ablation flag (off in Exp 09–10 to preserve v1.4 failure docs)
- **Exp 12** — prevents Exp 09-style misleading context without sleep
- **Exp 13** — flags ambiguous hypothesis competition on near-duplicate concepts

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
pytest tests/                      # 70 unit tests
python run_all_experiments.py      # All 23 experiments + summary
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
python experiments/exp_12_meta_misleading_context/run.py  # v3.0 A: meta detects decoy context
python experiments/exp_13_meta_ambiguous_reasoning/run.py # v3.0 B: ambiguous reasoning flags
python experiments/exp_14_meta_consequential/run.py       # v3.1: defer/sleep beats commit
python experiments/exp_15_active_epistemic_probe/run.py   # v4.0: epistemic probing
python experiments/exp_16_active_inference_ablation/run.py  # v4.0: active vs passive ablation
python experiments/exp_17_text_ambiguous_deferral/run.py    # v5.0: goal-directed text probe
python experiments/exp_18_text_session_memory/run.py        # v5.0: text + sleep recovery
python experiments/exp_19_hybrid_spike/run.py                 # v5.1: hybrid LLM gate
python experiments/exp_20_unified_gate/run.py                 # v6.0: unified gate
python experiments/exp_21_sbert_embeddings/run.py             # v6.0: SBERT embeddings
python experiments/exp_22_end_to_end/run.py                     # v6.2: full-stack E2E
python experiments/exp_23_eidos_eval/run.py                     # v7.0: EIDOS-Eval
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
- **Not an LLM replacement.** It grounds phrases to vectors and gates LLM drafts; it does not train on or predict tokens at scale.
- **Not biologically accurate.** It is inspired by neuroscience, not simulating neurons.
- **Not production-ready.** No API, no deployment, no scaling claims.

EIDOS is a tool for understanding cognition computationally — a Kisamapa Labs experiment in building intelligence from first principles.

## Future Expansion

- **Learned gate policy** — tune thresholds from benchmark logs
- **Richer grounding** — domain-specific embeddings beyond hash/SBERT

## Hybrid Spike (LLM + EIDOS)

Optional demo — LLM generates, EIDOS gates (`demos/hybrid_qa/`):

```bash
py demos/hybrid_qa/run.py                    # mock LLM
pip install -r requirements-hybrid.txt      # for GPT-2 CPU
py demos/hybrid_qa/run.py --gpt2
py experiments/exp_19_hybrid_spike/run.py   # measurable gate vs blind LLM
```


---

*KISAMAPA LABS — EXPERIMENT 06 — EIDOS v7.1*
*Classification: Research Prototype — Active*

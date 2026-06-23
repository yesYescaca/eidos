# EIDOS Sidecar: Augmenting LLM Calibration with Biologically-Grounded Cognitive Monitoring

**Francisco — Kisamapa Labs**  
**Repository:** https://github.com/yesYescaca/eidos  
**Version evaluated:** EIDOS v7.6 (June 2026)  
**Status:** Draft for workshop / arXiv preprint

---

## Abstract

Large language models (LLMs) often answer confidently on questions that invite false beliefs or lack sufficient context. Chain-of-thought (CoT) prompting is widely used to improve reasoning, but can amplify misconception traps on benchmarks such as TruthfulQA. We present **EIDOS Sidecar**, a hybrid architecture in which a numpy-only cognitive monitor—implementing predictive processing, metacognitive gating, and symbol grounding—wraps a commercial LLM. The monitor injects structured belief state (concept rankings, surprise, ambiguity flags) before generation and can withhold or clarify outputs when alignment checks fail. On TruthfulQA Misconceptions (N=50) and a mixed benchmark of 25 misconception plus 25 ambiguous items (N=50), evaluated across three Groq-hosted models, EIDOS belief injection improves **task accuracy** by 26–50 points versus LLM-alone on the mixed benchmark for all models tested. Against CoT, belief injection yields substantially higher commit accuracy on large Llama and GPT-OSS models where CoT degrades performance; on Llama-3.1-8B, CoT remains stronger on misconception-only commits while EIDOS still wins on the combined mixed task through superior ambiguity handling. We report results honestly, including abstention trade-offs on must-answer benchmarks, and release code, eval harness, and per-model reports.

**Keywords:** LLM calibration, metacognition, hybrid AI, TruthfulQA, chain-of-thought, cognitive architecture

---

## 1. Introduction

### 1.1 Problem

Fluent language generation does not imply reliable epistemic behaviour. LLMs frequently:

1. **Commit to misconceptions** — repeating popular falsehoods from training data (Lin et al., 2022).
2. **Over-commit under ambiguity** — answering when clarification is appropriate.

Standard mitigations include RLHF, uncertainty estimation, and chain-of-thought prompting (Wei et al., 2022). CoT improves performance on many reasoning tasks but is not uniformly beneficial: on TruthfulQA misconception items, extended reasoning can reinforce trap answers rather than correct them.

### 1.2 Approach

**EIDOS** (Emergent Intelligence via Distributed Organisational Systems) is a laboratory cognitive architecture built from neuroscience-inspired primitives—predictive coding, global workspace broadcasting, Hebbian learning, and metacognitive monitoring—without training a language model. **EIDOS Sidecar** (v5–v7) couples this monitor to an external LLM in a dual-process pattern inspired by Kahneman’s System 1 / System 2 distinction:

- **System 1:** LLM proposes a draft answer.
- **System 2:** EIDOS evaluates the draft against grounded concept state and may **commit**, **clarify**, **defer**, or **probe**.

The novel comparison in this work is **belief injection** versus **chain-of-thought**: belief carries structured signals from a separate cognitive architecture, not merely additional prompt tokens asking the model to “think step by step.”

### 1.3 Contributions

1. **Architecture:** A reproducible hybrid pipeline (`HybridEidosAgent`) with unified `GatePolicy`, SBERT grounding, and belief-context injection.
2. **Evaluation harness:** EIDOS-Eval with five modes (LLM-alone, CoT, gate-only, belief, meta-injection), TruthfulQA-aligned grading, and a mixed misconception+ambiguous benchmark.
3. **Empirical results:** N=50 live API runs on three Groq models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-20b`).
4. **Honest analysis:** Model-dependent effects, abstention trade-offs, and limitations of reference-based grading.

---

## 2. Related Work

| Area | Key references | Relation to EIDOS |
|------|----------------|-------------------|
| TruthfulQA | Lin et al. (2022) | Primary misconception benchmark; TI metric |
| LLM calibration | Kadavath et al. (2022) | Motivates selective commit vs coverage |
| Chain-of-thought | Wei et al. (2022) | Main baseline |
| Predictive processing | Friston (2010, 2017) | Surprise / prediction-error signals in gate |
| Metacognition | Nelson & Narens (1990) | Monitor-and-control over LLM drafts |
| Dual-process | Kahneman (2011) | Fast LLM + slow cognitive veto |
| Symbol grounding | Harnad (1990) | Text→vector bridge for concept alignment |

EIDOS Sidecar is **not** claimed to replace scaling laws or match SOTA on broad knowledge benchmarks. The claim is narrower: **improved calibration on misconception and ambiguity under a fixed LLM API.**

---

## 3. Method

### 3.1 Cognitive core (summary)

The EIDOS **Predictive Active Workspace (PAW)** maintains:

- A **prediction engine** computing surprise on incoming text.
- A **belief graph** and **episodic buffer** updated via Hebbian association.
- **Meta-cognition flags** (e.g. `ambiguous_hypothesis`, `low_confidence`).
- An **active inference** controller selecting observe / probe / sleep actions.

For Sidecar evaluation, text is grounded through `TextGroundingBridge` with **SBERT** embeddings (`all-MiniLM-L6-v2`), mapping user questions and domain concepts into a shared vector space.

### 3.2 Hybrid pipeline

```
User question
    → EIDOS step (surprise, concepts, meta flags)
    → [optional] Belief context block prepended to LLM prompt
    → LLM draft generation
    → GatePolicy.evaluate(draft ↔ goal, concept ambiguity, underdetermination)
    → commit | clarify | defer | probe
    → [optional] Meta-injection revision loop
    → Final response
```

**Belief context** (`build_belief_context`) includes:

- Top concept alignments for the question (ranked scores).
- Surprise ratio from the prediction engine.
- Meta-cognition monitor flags.
- Mode-specific instructions (`factual_mode` for misconception items: answer directly; default: clarify if concepts are close).

**GatePolicy** fuses cognitive decisions with text-alignment checks: draft–goal cosine similarity, concept ambiguity (top-two concepts within ε), draft–concept mismatch, and (v7.5) underdetermination markers for abstention on genuinely ambiguous phrasing only.

### 3.3 Eval modes (EIDOS-Eval)

| Mode | Gate | Belief injection | Meta revision | CoT |
|------|------|------------------|---------------|-----|
| `llm_alone` | off | off | off | off |
| `llm_cot` | off | off | off | on (“think step by step”) |
| `eidos_gate` | on | off | off | off |
| `eidos_belief` | on | on | off | off |
| `eidos_meta` | on | on | on | off |

Each question includes registered **concepts** (truth/falsehood phrases) and optional **warmup** training steps so the monitor encodes domain structure before the LLM is called.

### 3.4 Benchmarks

#### TruthfulQA Misconceptions (N=50)

- Subset of Lin et al. (2022) *Misconceptions* category, built via `build_truthfulqa_subset.py`.
- **Must-answer** setting: abstention counts against task score.
- **Metrics (Lin et al.):**
  - **T** — truthful (no false reference matched; abstention counts as truthful).
  - **I** — informative (substantive committed answer).
  - **TI** — truthful **and** informative (headline).
  - **misc** — committed answer matches a known incorrect reference.
  - **commit TI** — TI rate among committed answers only.

Grading uses reference answer lists (substring/token overlap), not GPT-judge or BLEURT (see limitations).

#### Mixed benchmark (N=50)

- 25 misconception items (must answer correctly) + 25 ambiguous items (must abstain or clarify).
- Built via `build_mixed_eval_50.py`.
- **task_accuracy** — correct commits on misconceptions + correct abstentions on ambiguous items.
- **ambig_safe_rate** — no false commit on ambiguous-labeled items.
- **misconception_commit_ti_rate** — TI on misconception items when committing.

This benchmark matches the design goal of EIDOS: **know when to answer and when not to.**

### 3.5 Experimental setup

| Parameter | Value |
|-----------|-------|
| Provider | Groq API |
| Models | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-20b` |
| Embedding | SBERT (auto fallback to hash if HF unavailable) |
| Temperature | 0.2 |
| Max tokens | 256 |
| N per benchmark | 50 |
| Seed | 42 |
| Gate profile | `LIVE_TRUTHFULQA_V75` (factual_mode, underdetermination-gated abstention) |

Raw JSON reports: `eval/eidos_eval/reports/live_{benchmark}_{model}_report.json`.

---

## 4. Results

### 4.1 TruthfulQA Misconceptions (N=50)

#### Table 1 — `llama-3.3-70b-versatile`

| Mode | TI | misc | Abstain | Commit TI |
|------|-----|------|---------|-----------|
| LLM alone | **88%** | 4% | 0% | 88% |
| LLM CoT | 62% | 4% | 0% | 62% |
| EIDOS gate | 86% | 2% | 4% | 89.6% |
| EIDOS belief | 82% | 2% | 6% | **87.2%** |
| EIDOS meta | 82% | 2% | 6% | 87.2% |

**Belief vs CoT (commit TI):** +25.2 pts (87.2% vs 62%).  
**Belief vs alone (TI):** −6 pts — cost of appropriate abstention on ~6% of items.  
**Misconception commits halved:** misc 4% → 2%.

#### Table 2 — `llama-3.1-8b-instant`

| Mode | TI | misc | Abstain | Commit TI |
|------|-----|------|---------|-----------|
| LLM alone | 74% | 6% | 0% | 74% |
| LLM CoT | **90%** | 6% | 0% | **90%** |
| EIDOS gate | 64% | 4% | 16% | 76.2% |
| EIDOS belief | 66% | 6% | 12% | 75.0% |
| EIDOS meta | 68% | 6% | 10% | 75.6% |

**Belief vs CoT (commit TI):** −15 pts — CoT helps the small model on factual traps.  
EIDOS abstains more (12–16%), hurting must-answer TI.

#### Table 3 — `openai/gpt-oss-20b`

| Mode | TI | misc | Abstain | Commit TI |
|------|-----|------|---------|-----------|
| LLM alone | 52% | 2% | 0% | 52% |
| LLM CoT | 36% | 0% | 0% | 36% |
| EIDOS gate | 48% | 0% | 34% | 72.7% |
| EIDOS belief | 54% | 0% | 24% | **71.1%** |
| EIDOS meta | 56% | 0% | 18% | 68.3% |

**Belief vs CoT (commit TI):** +35.1 pts (71.1% vs 36%).  
**Belief vs alone (TI):** +2 pts despite 24% abstention.

#### Summary — TruthfulQA

| Model | Belief beats CoT (commit TI)? | Belief beats alone (TI)? |
|-------|------------------------------|--------------------------|
| Llama-3.3-70B | **Yes** (+25 pts) | No (−6 pts) |
| Llama-3.1-8B | No (−15 pts) | No (−8 pts) |
| GPT-OSS-20B | **Yes** (+35 pts) | **Yes** (+2 pts) |

**Interpretation:** CoT is harmful on strong/large models and weak OSS models for misconception traps—exactly where belief injection helps. On 8B, CoT provides useful scaffolding; EIDOS gate thresholds may over-abstain.

---

### 4.2 Mixed benchmark (N=50) — primary result

#### Table 4 — `llama-3.3-70b-versatile`

| Mode | Task acc | Ambig safe | Miscon commit TI |
|------|----------|------------|------------------|
| LLM alone | 60% | 28% | 92% |
| LLM CoT | 40% | 20% | 60% |
| EIDOS belief | 86% | 88% | 84% |
| EIDOS gate | **96%** | **100%** | 92% |
| EIDOS meta | 86% | 88% | 84% |

**Belief vs alone (task):** +26 pts. **Belief vs CoT (task):** +46 pts.

#### Table 5 — `llama-3.1-8b-instant`

| Mode | Task acc | Ambig safe | Miscon commit TI |
|------|----------|------------|------------------|
| LLM alone | 56% | 28% | 84% |
| LLM CoT | 66% | 40% | 92% |
| EIDOS belief | 86% | **100%** | 78.3% |
| EIDOS gate | 88% | 96% | 83.3% |
| EIDOS meta | 86% | 100% | 78.3% |

**Belief vs alone (task):** +30 pts. **Belief vs CoT (task):** +20 pts.  
CoT still wins on misconception commits (92% vs 78%), but EIDOS wins overall via ambiguity safety (100% vs 40%).

#### Table 6 — `openai/gpt-oss-20b`

| Mode | Task acc | Ambig safe | Miscon commit TI |
|------|----------|------------|------------------|
| LLM alone | 30% | 8% | 52% |
| LLM CoT | 12% | 0% | 24% |
| EIDOS belief | 80% | **100%** | 75% |
| EIDOS gate | 72% | 92% | 68.4% |
| EIDOS meta | 80% | 96% | 72.7% |

**Belief vs alone (task):** +50 pts. **Belief vs CoT (task):** +68 pts.

#### Summary — Mixed (EIDOS belief)

| Model | Δ task vs alone | Δ task vs CoT | Ambig safe (belief) | Belief beats CoT (miscon commits)? |
|-------|-----------------|---------------|---------------------|-----------------------------------|
| Llama-3.3-70B | +26 pts | +46 pts | 88% | **Yes** |
| Llama-3.1-8B | +30 pts | +20 pts | 100% | No |
| GPT-OSS-20B | +50 pts | +68 pts | 100% | **Yes** |

**Key finding:** On the benchmark aligned with EIDOS’s design goal, **belief injection improves task accuracy on all three models**, with ambiguity-safe rates of 88–100% versus 8–40% for LLM-alone.

---

## 5. Discussion

### 5.1 Why mixed results matter more than TruthfulQA-only

TruthfulQA in our harness penalizes appropriate abstention: a system that correctly withholds on uncertain items loses TI. EIDOS’s gate is intentionally conservative. The **mixed benchmark** rewards both factual commits and ambiguity awareness; here EIDOS shows its largest gains (+26 to +50 task accuracy points).

### 5.2 Belief vs CoT — when does each win?

| Condition | Observed pattern |
|-----------|------------------|
| Large Llama (70B) | CoT hurts on misconceptions; belief helps |
| Small Llama (8B) | CoT helps misconceptions; belief wins on mixed via abstention |
| Weak OSS (20B) | CoT catastrophic; belief strongly preferred |

This suggests EIDOS Sidecar is most valuable when (a) the base model is prone to confident errors, or (b) the task requires **calibrated commit/abstain** behaviour—not merely more internal monologue.

### 5.3 Gate vs belief

Gate-only (`eidos_gate`) sometimes outperforms belief on mixed task accuracy (e.g. 96% vs 86% on 70B) by abstaining more aggressively. Belief injection trades some abstention precision for richer LLM conditioning. Meta-injection adds revision rounds with diminishing returns on these benchmarks.

### 5.4 Limitations

1. **N=50 per benchmark per model** — directional, not definitive; no confidence intervals reported.
2. **Single API provider (Groq)** — results may vary by hosting and snapshot.
3. **Reference-based grading** — not official TruthfulQA GPT-judge; may diverge from leaderboard metrics.
4. **Concept warmup required** — eval items include per-question concept registration; cold-start deployment is untested.
5. **No fine-tuning** — all gains from inference-time architecture, not weight updates.
6. **8B regression** — belief does not beat CoT on misconception-only commits; gate calibration for small models is future work.
7. **Latency & cost** — multiple EIDOS steps + optional revision loops increase inference cost versus raw LLM.

### 5.5 Threats to validity

- **Cache effects:** Per-model response caches may stabilize but also freeze early errors; `--no-cache` ablation recommended for final publication.
- **Subset selection:** N=50 TruthfulQA subset is seeded but not identical to full 104-item misconception set.
- **Prompt sensitivity:** CoT and belief prompts were hand-designed; modest paraphrase sensitivity is likely.

---

## 6. Conclusion

EIDOS Sidecar demonstrates that a lightweight, interpretable cognitive monitor can improve LLM behaviour on misconception and ambiguity benchmarks without retraining the language model. The strongest empirical claim is on the **mixed N=50 benchmark**, where belief injection raises task accuracy by 26–50 points across three Groq models while achieving 88–100% ambiguity-safe behaviour. Against chain-of-thought, belief injection wins decisively on Llama-3.3-70B and GPT-OSS-20B for misconception commit accuracy; on Llama-3.1-8B, CoT remains competitive on facts while EIDOS wins on the combined task.

Future work includes: full 104-item TruthfulQA misconceptions, GPT-judge grading replication, per-model gate calibration, latency-optimized deployment, and ablations isolating belief injection vs gate-only vs meta-injection.

---

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.
- Friston, K. et al. (2017). Active inference: a process theory. *Neural Computation*.
- Harnad, S. (1990). The symbol grounding problem. *Physica D*.
- Kadavath, S. et al. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Lin, S. et al. (2022). TruthfulQA: Measuring how models mimic human falsehoods. *ACL 2022*.
- Nelson, T. O., & Narens, L. (1990). Metamemory: A theoretical framework and new findings. *Psychology of Learning and Motivation*.
- Wei, J. et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 2022*.

---

## Appendix A — Development methodology notes

Early live evals (v7.3) used a harsh substring scorer and counted all abstention as failure on must-answer items, producing misleading 4–16% accuracy. v7.4 introduced TruthfulQA-aligned TI grading; v7.5 calibrated abstention via underdetermination markers (`underdetermination.py`) and introduced the mixed benchmark. v7.6 added multi-model support. This progression is documented in `docs/LIVE_EVAL_PILOT.md` and `CHANGELOG.md`.

## Appendix B — Reproduction commands

```bash
pip install -r requirements.txt -r requirements-eval.txt
set GROQ_API_KEY=...

# TruthfulQA N=50
py -m eval.eidos_eval.live_runner --provider groq --model llama-3.3-70b-versatile --truthfulqa
py -m eval.eidos_eval.live_runner --provider groq --model llama-3.1-8b-instant --truthfulqa
py -m eval.eidos_eval.live_runner --provider groq --model openai/gpt-oss-20b --truthfulqa

# Mixed N=50
py -m eval.eidos_eval.live_runner --provider groq --model llama-3.3-70b-versatile --mixed
py -m eval.eidos_eval.live_runner --provider groq --model llama-3.1-8b-instant --mixed
py -m eval.eidos_eval.live_runner --provider groq --model openai/gpt-oss-20b --mixed

# All models batch
py -m eval.eidos_eval.run_multimodel_eval --provider groq
```

## Appendix C — Figure suggestions (for LaTeX submission)

1. **Bar chart:** Mixed task accuracy — belief vs CoT vs alone (3 models, grouped bars).
2. **Bar chart:** Ambiguity-safe rate on mixed benchmark (belief vs alone).
3. **Diagram:** Hybrid pipeline (Section 3.2 flow).
4. **Table:** Summary of Table 4–6 in single cross-model view.

## Appendix D — Data availability

- Code and eval harness: https://github.com/yesYescaca/eidos (tag `v7.6`)
- Local reports (not in git): `eval/eidos_eval/reports/*.json`
- Question sets: `questions_truthfulqa_50.json`, `questions_mixed_50.json`

---

*Draft generated from EIDOS v7.6 live eval reports. Revise author affiliation, add statistical tests, and convert to LaTeX before submission.*

# Changelog

## v7.8 — Report Stats, N=104 Wiring, Paper Ablation
- **`eval/eidos_eval/stats.py`** — Wilson CIs, paired bootstrap, McNemar exact (stdlib only)
- **`analyze_reports.py`** — CLI over `live_*_report.json`; writes `stats_summary.json`
- **`--truthfulqa-full`** — TruthfulQA Misconceptions N=104 (`questions_truthfulqa_104.json`)
- **`run_multimodel_eval`** — optional benchmark key `truthfulqa_full`
- **`docs/EIDOS_Research_Paper.html`** — §4.5 Table 10 mode ablation; Wilson CIs in Table 7; Appendix D updated
- **`docs/PAPER_EVAL_COMMANDS.md`** — N=104 + stats CLI sections
- **Exp 30** — report stats smoke test
- **`tests/test_stats.py`** — Wilson + bootstrap unit tests
- State version **7.8**

## v7.7 — Reflection Baseline
- **`EvalMode.llm_reflection`** — two-call self-critique baseline (draft → revise)
- **`reflection.py`** — `run_reflection_baseline()` separate from EIDOS gate
- **`run_live_eval.py`** — repo-root launcher (fixes wrong-cwd `ModuleNotFoundError`)
- **Comparison summary** — `belief_beats_reflection`, reflection TI deltas
- **`GROQ_EXTENDED_EVAL_MODELS`** — GPT-OSS-120B, Qwen3.6-27B, Llama 4 Scout
- **`run_multimodel_eval --extended`** — batch 6-model eval
- **`docs/PAPER_EVAL_COMMANDS.md`** — reproduction commands + live result tables
- **`docs/EIDOS_Research_Paper.html`** — preprint with N=50 results (6 models on mixed)
- **Exp 29** — reflection mock CI
- Live runner includes reflection in default comparison modes
- State version **7.7**

## v7.6 — Multi-Model Live Eval
- **`--model`** on live runner — explicit Groq/OpenAI model ID
- **Per-model cache** — `live_cache_{model_slug}.json`
- **`run_multimodel_eval.py`** — TruthfulQA + mixed across default Groq model set
- **`live_models.py`** — `GROQ_EVAL_MODELS` (70b Llama, 8b, GPT-OSS-20B)
- **Reports** — `eval/eidos_eval/reports/` per model + benchmark
- **`docs/MULTIMODEL_EVAL.md`** — multi-model workflow
- **Exp 28** — model registry smoke test
- State version **7.6**

## v7.5 — Abstention Calibration + Mixed Benchmark
- **`LIVE_TRUTHFULQA_V75`** — underdetermination-gated abstention; truth-concept commit
- **`questions_mixed_50.json`** — 25 misconceptions + 25 ambiguous (`build_mixed_eval_50.py`)
- **Subset metrics** — `misconception_commit_ti_rate`, `belief_beats_cot_misconception_commits`
- **`--mixed`** on live runner
- **Exp 27** — mixed eval mock CI
- State version **7.5**

## v7.4 — TruthfulQA-Aligned Grading + Factual Gate
- **`truthfulqa_scorer.py`** — T / I / TI metrics per Lin et al. (ACL 2022)
- **Reference answer lists** in `questions_truthfulqa_50.json` (rebuild via builder)
- **`LIVE_TRUTHFULQA`** gate profile + `factual_mode` on `GatePolicy`
- **Factual belief prompt** — direct answers on misconception questions
- **Exp 26** — TruthfulQA grading mock CI
- Live runner reports T/I/TI + saves `live_truthfulqa_report.json`
- State version **7.4**

## v7.3 — TruthfulQA N=50 + CoT Baseline + Gate Calibration
- **`questions_truthfulqa_50.json`** — 50 TruthfulQA Misconceptions (`build_truthfulqa_subset.py`)
- **`EvalMode.llm_cot`** — chain-of-thought baseline vs EIDOS belief
- **`gate_profiles.py`** + **`calibrate_gate.py`** — empirical gate-only tuning
- **`docs/LIVE_EVAL_PILOT.md`** — honest N=6 Groq pilot results
- **`docs/SIDECAR_RESEARCH_NOTE.md`** — arXiv-ready research arc
- **Exp 25** — TruthfulQA meta vs CoT
- **`--truthfulqa`** on live runner
- State version **7.3**

# v7.2 — Calibrated live Sidecar + task accuracy metrics
- **Live eval uses SBERT** + `GATE_LIVE_MIN_DRAFT_GOAL_ALIGN=0.72` (fixes 100% over-abstention)
- **Gate calibration** — question-clarity bypass; concept mismatch only when draft–goal below threshold
- **`task_accuracy`** + **`ComparisonSummary`** selective accuracy deltas
- **`EvalMode.eidos_belief`** — gate + belief-grounded prompts
- **Belief context in meta-injection revisions**
- **`CachedLLM`** — disk cache for live API responses (`live_cache.json`)
- **LLM retry** on 429/5xx with backoff
- State version **7.2**

## v7.1 — Live Groq Eval + Belief-Grounded Sidecar
- **`create_live_llm("groq")`** — Groq OpenAI-compatible API (`GROQ_API_KEY`, `llama-3.3-70b-versatile` default)
- **`eval/eidos_eval/live_runner.py`** — live API comparison CLI (skips without API key)
- **`questions_live.json`** — 6-item live subset (TruthfulQA-inspired + domains)
- **Belief-grounded prompts** — `enable_belief_context` injects EIDOS concept rankings before LLM generation
- **`GatePolicy` draft–concept mismatch** — veto when draft best-matches wrong concept
- **Exp 24** — optional live Groq eval (CI-safe skip)
- **`demos/hybrid_qa`** — `--groq`, `--meta-injection`, `--belief-context`
- State version **7.1**

## v7.0 — EIDOS-Eval + Metacognitive Injection
- **`eval/eidos_eval/`** — graded questions, LLM-alone vs gate vs meta comparison harness
- **`OpenAICompatibleLLM`** — optional API backend (stdlib urllib, `OPENAI_API_KEY`)
- **`RoundRobinMockLLM`** — CI-safe revision simulation
- **Metacognitive injection** — `enable_meta_injection` revision loop in `HybridEidosAgent`
- **`create_hybrid_grounding()`** — SBERT default for hybrid path, hash fallback
- **`GatePolicy`** — promote aligned drafts to commit after concept-ambiguity clarify
- **`docs/POSITIONING.md`** — EIDOS Core vs Sidecar narrative
- **Exp 23** — EIDOS-Eval proves gate reduces false commits vs LLM-alone
- State version **7.0**

## v6.2 — Real-World Benchmark Expansion
- **`cases.json` v2.0** — 12 real-world cases across IT, support, security, finance, clinical, HR, legal, aviation, education, logistics
- **Per-category/domain metrics** in `AmbiguousQABenchmark` (`by_category`, `by_domain`, filters)
- **`benchmark/ambiguous_qa/README.md`** — domain index and references
- **Docs polish** — README, LAB_REPORT, synthesis updated for v6.2

## v6.1 — Ambiguous QA Benchmark + End-to-End Exp 22
- **`benchmark/ambiguous_qa/`** — labeled cases, `AmbiguousQABenchmark` runner, metrics
- **Exp 22** — misleading context → sleep → full stack (meta + active + gate) vs baseline
- **`HybridEidosAgent.respond(reset=...)`** — session-preserving respond for multi-phase flows
- **`CaseMockLLM`** — per-case deterministic LLM drafts for benchmark

## v6.0 — Unified Gate + Semantic Embeddings
- `GatePolicy` — fuses cognitive steps, draft↔goal alignment, concept ambiguity
- `gate_response()` + `GateEvaluation` audit trail (`scores`, `reasons`)
- `HybridEidosAgent.use_unified_gate` (default on); legacy merge via `use_unified_gate=False`
- `create_grounding("hash" | "sbert")` + `SentenceTransformerGrounding` (optional)
- Exp 20–21: unified gate vs legacy merge; SBERT separation (optional dep)
- State version **6.0**

## v5.1 — Hybrid Spike (LLM + EIDOS)
- `HybridEidosAgent` — LLM proposes, EIDOS gates output
- `MockLanguageModel` + optional `GPT2LanguageModel` (CPU)
- Exp 19 + `demos/hybrid_qa/`
- See `docs/HYBRID_SPIKE_PLAN.md` for v6 roadmap

## v5.0 — Language Grounding Bridge
- `TextGroundingBridge` — deterministic hash embeddings (numpy only)
- `EidosTextAgent` — `register_text_concept`, `step_text`, `text_decision`
- Exp 17–18: goal-directed text probe + text session memory via sleep

## v4.0 — Active Inference
- `ActiveInferenceController` — expected free energy action selection (Friston 2017)
- Actions: `observe`, `probe:<concept>`, `sleep`
- `enable_active_inference` flag (default off for Exp 01–14 compatibility)
- Exp 15–16: epistemic probing and ablation

## v3.1 — Consequential Meta-Cognition
- `enable_meta_consequential` flag (default on)
- Defer hypothesis when `ambiguous_hypothesis` or `low_confidence` (v3.0 required both)
- Auto `sleep()` on misleading context and on ambiguous reasoning (retry)
- Exp 14: v3.1 beats v3.0 commit on ambiguous recovery

## v3.0 — Meta-Cognition
- `MetaCognitionMonitor`: misleading context detection + reasoning flags
- Exp 12–13; `enable_meta_cognition` ablation flag

## v2.0 — Complementary Learning Systems
- `EpisodicBuffer`, `BeliefGraph`, `SleepReplay`, `agent.sleep()`
- Exp 11 fixes Exp 09/10 failure modes via sleep

## v1.4 — Autonomous Recovery
- `RecoveryContextTracker`; Exp 08

## v1.3 — Consolidation Preview
- Multi-concept disambiguation without hardcoded overrides; Exp 07

## v1.2 — Belief Consolidation
- Nonlinear MLP; `consolidate_belief()`; Exp 06

## v1.1 — Closed-Loop Reasoning
- Relative surprise; hypothesis feedback; Exp 04–05

## v1.0 — Initial PAW
- Full architecture; Exp 01–03

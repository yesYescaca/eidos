# Changelog

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

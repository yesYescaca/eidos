# Hybrid Spike — LLM + EIDOS (System 1 + System 2)

## Goal

Demonstrate **enhancement, not replacement**: a small LLM generates text; EIDOS monitors surprise, memory, and meta-cognition and **gates** the output.

## Architecture

```
User text
    ↓
[EidosTextAgent]  monitor question (surprise, context, active inference)
    ↓
[LLM backend]     draft answer (GPT-2 CPU or mock for tests)
    ↓
[EidosTextAgent]  monitor draft vs goal / concepts
    ↓
[HybridGate]      commit | defer | clarify | probe
    ↓
Final response
```

**LLM = mouth (fluency). EIDOS = judgment (when to speak, defer, or probe).**

## Research touchpoints

- Dual-process theory (Kahneman) — fast generation vs slow monitoring
- Active inference — probe before committing under uncertainty
- Metacognition — defer when reasoning is ambiguous
- Clark (2013) — language as scaffolding for predictive cognition

## Scope (spike only)

| In scope | Out of scope (v6+) |
|----------|-------------------|
| `HybridEidosAgent` + gate policy | Unified v6 policy learner |
| Mock LLM for Exp 19 (CI-safe) | Fine-tuning GPT-2 |
| Optional GPT-2 demo script | Production API |
| Exp 19: gate beats blind commit on ambiguity | Full chat product |

## v6 path (future)

1. **Unified gate** — single `GatePolicy` combining meta + EFE + text similarity
2. **Richer embeddings** — optional sentence-transformers backend
3. **Session API** — `HybridEidosAgent.respond()` as library entry point
4. **Benchmark** — ambiguous QA set with defer/commit labels

## Dependencies

Core EIDOS unchanged. Hybrid adds optional `requirements-hybrid.txt`:

- `torch` (CPU)
- `transformers`

Tests and Exp 19 use **MockLLM** only — no download required.

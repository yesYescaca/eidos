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

| In scope | Out of scope |
|----------|-------------|
| `HybridEidosAgent` + gate policy | Fine-tuning GPT-2 |
| Mock LLM for Exp 19 (CI-safe) | Production API |
| Optional GPT-2 demo script | Full chat product |
| Exp 19: gate beats blind commit on ambiguity | Learned gate policy |

## v6.0 (shipped)

1. **`GatePolicy`** — unified cognitive + draft↔goal + concept ambiguity (`architecture/gate/`)
2. **`create_grounding("hash" \| "sbert")`** — optional sentence-transformers (`requirements-embeddings.txt`)
3. **Exp 20–21** — legacy merge vs unified gate; SBERT separation benchmark

## v6.1–v6.2 (shipped)

1. **`benchmark/ambiguous_qa/`** — 17 labeled cases (lab + real-world professional domains)
2. **Exp 22** — end-to-end full stack vs baseline; benchmark metrics in results
3. **`respond(reset=False)`** — session-preserving hybrid flows

## Future

1. **Learned gate thresholds** from benchmark logs (`benchmark/ambiguous_qa/`)
2. **Belief-grounded LLM context** — retrieve consolidated beliefs instead of raw chat

## Dependencies

Core EIDOS unchanged. Hybrid adds optional `requirements-hybrid.txt`:

- `torch` (CPU)
- `transformers`

Tests and Exp 19 use **MockLLM** only — no download required.

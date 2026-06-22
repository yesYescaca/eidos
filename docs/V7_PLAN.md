# EIDOS v7.0 — EIDOS-Eval + Metacognitive Injection

## Problem (post-v6.2)

- Benchmark proved **internal** gate safety only (gate vs no gate on labeled cases).
- No **external** comparison: LLM-alone vs LLM+EIDOS on graded correctness.
- Gate blocked output but did not **feed cognitive state back** into the LLM for revision.

## v7.0 Solution

### 1. Dual positioning (`docs/POSITIONING.md`)

- **EIDOS Core** — numpy cognitive lab (v1–v4)
- **EIDOS Sidecar** — LLM monitor + gate + optional meta-injection (v5+)

### 2. Metacognitive injection

`HybridEidosAgent.respond(..., enable_meta_injection=True)`:

1. LLM generates draft
2. Gate evaluates → if not `commit`, inject monitor block into prompt
3. LLM revises (up to `META_INJECTION_MAX_ROUNDS`)
4. Re-gate revised draft

Inspired by Nelson & Narens (1990) metacognitive control and deliberation loops.

### 3. EIDOS-Eval (`eval/eidos_eval/`)

Graded question set + harness comparing:

| Mode | Description |
|------|-------------|
| `llm_alone` | Blind commit of LLM draft |
| `eidos_gate` | Unified gate — abstain/clarify/defer |
| `eidos_meta` | Gate + metacognitive revision loop |

Metrics: accuracy, accuracy-when-commit, abstention rate, false-commit rate, selective accuracy delta.

### 4. SBERT default for hybrid path

`create_hybrid_grounding()` — tries SBERT, falls back to hash (CI-safe).

### 5. Exp 23

EIDOS-Eval on mock LLM (CI) proves Sidecar reduces false commits and meta-injection improves committed accuracy.

## References

- Nelson & Narens (1990) — metacognitive monitoring and control
- Kadavath et al. (2022) — language model calibration
- Yin et al. (2023) — uncertainty in LLMs

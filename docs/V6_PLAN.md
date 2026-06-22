# EIDOS v6.0 — Unified Gate Policy

## Problem (v5.1)

Gate logic was split across:
- `interpret_text_decision()` per cognitive step
- `merge_decisions()` in hybrid agent
- No **draft vs goal** text alignment check
- `observe` still passed through wrong LLM drafts

## v6.0 Solution: `GatePolicy`

Single module fusing:
1. **Cognitive signals** — meta flags, active inference, hypothesis deferral
2. **Text alignment** — draft ↔ goal similarity
3. **Concept ambiguity** — close top-2 phrase matches on user input

Output: `GateEvaluation(decision, scores, reasons)` with full audit trail.

## Experiments

| # | Claim |
|---|--------|
| 20 | Unified gate catches draft–goal misalignment when v5.1 piecemeal gate would pass draft |
| 21 | SBERT embeddings improve similar/unrelated separation vs hash (optional dep) |

## Embedding upgrade (v6.0)

- `create_grounding("hash" | "sbert")` factory
- `SentenceTransformerGrounding` — MiniLM → 64-d projection (CPU)
- Core lab still runs on hash; SBERT optional via `requirements-embeddings.txt`

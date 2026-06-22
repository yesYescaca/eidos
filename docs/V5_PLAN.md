# EIDOS v5.0 — Language Grounding Bridge

## Research basis

- **Symbol grounding** (Harnad 1990) — symbols must connect to sensorimotor/perceptual representations
- **Distributional semantics** (Landauer & Dumais 1997) — similar meanings occupy similar vector spaces
- **Clark (2013)** — predictive brains; language as compressed context for inference
- **EIDOS constraint** — no GPU, no LLM training; bridge must run on laptop with numpy

## Design

### `TextGroundingBridge`

Deterministic **hash + random projection** embedding (no torch, no API):

- Tokenise text (words + character trigrams)
- Hash into sparse bucket vector
- Project to 64-d via fixed random matrix (seeded)
- L2-normalise → compatible with `EidosAgent.input_dim`

Similar phrases → similar vectors (for near-duplicate reasoning/meta tests).

### `EidosTextAgent`

Thin wrapper over `EidosAgent`:

- `register_text_concept(label, phrase)` — embed + `register_concept`
- `step_text(phrase, ...)` — embed + `step` + `text_decision`
- `text_decision`: `observe` | `probe` | `defer` | `clarify` | `commit` | `sleep`

### Integration

- Core agent unchanged; text is an **adapter layer**
- State version **5.0** stores text concept source phrases
- No new required pip dependencies

## Experiments

| # | Name | Claim |
|---|------|-------|
| 17 | Text ambiguous deferral | Similar phrases → meta defers on ambiguous text input |
| 18 | Text session memory | Text concepts + sleep recovers after misleading phrase context |

## Out of scope

- LLM generation
- sentence-transformers / GPU embeddings (optional future extra)
- Full dialogue system

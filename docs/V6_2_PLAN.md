# EIDOS v6.2 — Real-World Benchmark Expansion

## Problem (post-v6.1)

The ambiguous QA benchmark had 5 **lab-style** cases (reactor, building, medical).
No coverage of everyday professional scenarios where LLMs over-commit:
helpdesk triage, fraud vs duplicate charges, phishing vs marketing, etc.

## v6.2 Solution

### Expanded `cases.json` (v2.0)

| Category | Count | Examples |
|----------|-------|----------|
| `lab` | 5 | Existing reactor/building/medical toy domains |
| `real_world` | 12 | IT, support, security, HR, finance, clinical, aviation, legal, education, logistics |

Each real-world case includes:
- `category`: `lab` | `real_world`
- `domain`: professional area (e.g. `it_support`, `cybersecurity`)
- `source_note`: plain-language inspiration (not copied from any dataset)
- Near-duplicate concepts reflecting **real confusions** practitioners face

Inspired by:
- Metacognitive calibration (Nelson & Narens 1990)
- Epistemic uncertainty in expert decision-making (Klein 1998 naturalistic decision-making)
- LLM overconfidence under ambiguity (Yin et al. 2023; Kalai et al. 2025 on deferral)

### Runner enhancements

- Filter by `category` / `domain` / `tags`
- Per-category metrics: `lab_safe_rate`, `real_world_safe_rate`
- `BenchmarkReport.by_domain` breakdown

### Pass criteria (tests)

- All `must_gate` cases safe (no false commits) — lab + real_world
- Decision match rate ≥ 85% overall (probe allowed on clear cases)

## Out of scope (v6.2)

- External dataset imports (SQuAD, etc.) — cases are hand-authored for gate semantics
- Learned threshold tuning — v6.3
- OOD holdout split — v6.3

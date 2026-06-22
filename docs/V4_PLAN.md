# EIDOS v4.0 — Active Inference Plan

## Research basis

**Friston (2010)** — cognition minimises free energy via perception *or* action.  
**Friston et al. (2017)** — active inference: select actions that minimise **expected free energy (EFE)**.

EFE decomposes into:
- **Epistemic value** — resolve uncertainty (information gain)
- **Pragmatic value** — move toward preferred / goal states

EIDOS already minimises surprise by **updating beliefs** (prediction, reasoning, sleep). v4 adds **acting to sample the world** before committing.

## Minimal v4 implementation (laptop-friendly)

### New component: `ActiveInferenceController`

Discrete action space:

| Action | Meaning |
|--------|---------|
| `observe` | Process sensory input unchanged (v3 behaviour) |
| `probe:<concept>` | Actively sample a registered concept vector |
| `sleep` | Offline consolidation when probes are weak |

EFE (minimise):

```
G(a) = -(w_e * epistemic(a) + w_p * pragmatic(a))
```

- **Epistemic**: expected error reduction if action `a` is taken (`predict_no_learn` on probe vs observation)
- **Pragmatic**: cosine similarity of probed concept to current goal (if set)

### Integration

- Flag: `enable_active_inference` (default **False** — preserves Exp 01–14)
- Trigger when: unknown label + high error + (surprise spike **or** cold episodic context)
- Runs **before** weight-updating prediction in `step()`
- Returns `selected_action`, `expected_free_energy`, `action_epistemic_value` in step result

### Experiments

| # | Name | Claim |
|---|------|-------|
| 15 | Epistemic probe disambiguation | Active agent probes correct concept under ambiguous input |
| 16 | Active inference ablation | v4 on beats v4 off on cold-start identification with goal |

## Out of scope for v4.0

- Full generative model over hidden states
- Continuous action spaces
- LLM bridge (future track)

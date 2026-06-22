# Changelog

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

# EIDOS v7.5 — Abstention Calibration + Mixed Benchmark

## Goals (Claude feedback + v7.4 follow-up)

1. **Abstention calibration** — only gate/clarify when question has genuine underdetermination markers; factual drafts matching truth concept should commit
2. **Mixed N=50 benchmark** — 25 misconception (must answer) + 25 ambiguous (must abstain/clarify)
3. **Subset metrics** — misconception commit TI (belief vs CoT headline on commits only)
4. **Re-run path** — `--mixed` on live runner

## Deliverables

| Item | Path |
|------|------|
| Underdetermination gate | `architecture/gate/underdetermination.py`, `GatePolicy` v7.5 |
| Profile | `LIVE_TRUTHFULQA_V75` in `gate_profiles.py` |
| Mixed builder | `eval/eidos_eval/build_mixed_eval_50.py` |
| Question set | `questions_mixed_50.json` |
| Subset metrics | `runner.py`, `live_runner.py` |
| Exp 27 | `exp_27_mixed_eval/` |

## References

- Lin et al. (2022) — truth vs informativeness
- v7.4 N=50 Groq: belief TI 78% vs CoT 64%; alone 86%

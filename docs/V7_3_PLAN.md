# EIDOS v7.3 — TruthfulQA Scale + CoT Baseline + Gate Calibration

## Goals (Claude + pilot feedback)

1. **N=50 TruthfulQA Misconceptions** — meaningful percentages, maps to confident-wrong failure mode
2. **Gate-only calibration** — reduce over-abstention on clear questions (pilot: 66.7% abstain)
3. **Belief vs CoT baseline** — structured cognitive state vs "think step by step"
4. **Publish pilot N=6 Groq results** — honest framing in docs

## Deliverables

| Item | Path |
|------|------|
| TruthfulQA builder | `eval/eidos_eval/build_truthfulqa_subset.py` |
| N=50 question set | `eval/eidos_eval/questions_truthfulqa_50.json` |
| CoT prompts | `eval/eidos_eval/prompts.py` |
| Gate profiles | `architecture/gate/gate_profiles.py` |
| Calibrator | `eval/eidos_eval/calibrate_gate.py` |
| Exp 25 | TruthfulQA mock + belief vs CoT checks |
| Research note outline | `docs/SIDECAR_RESEARCH_NOTE.md` |

## References

- Lin et al. (2022) TruthfulQA
- Kadavath et al. (2022) calibration
- Wei et al. (2022) chain-of-thought prompting

# EIDOS-Eval (v7.4)

External comparison harness: **LLM-alone** vs **chain-of-thought** vs **EIDOS gate** vs **EIDOS belief** vs **EIDOS meta-injection**.

## Modes

| Mode | Gate | Belief context | Meta revision | CoT |
|------|------|----------------|---------------|-----|
| `llm_alone` | off | off | off | off |
| `llm_cot` | off | off | off | on |
| `eidos_gate` | on (calibrated) | off | off | off |
| `eidos_belief` | on | on | off | off |
| `eidos_meta` | on | on | on | off |

## Metrics

- **task_accuracy** — correct committed answers + correct abstentions on `must_abstain` items
- **accuracy** — fraction of questions with correct committed response
- **accuracy_when_commit** — correctness on committed answers only
- **abstention_rate** — fraction withheld (gated non-commit)
- **false_commit_rate** — wrong answer committed on `must_abstain` items
- **must_abstain_safe_rate** — no false commits on abstain-labeled items

Belief vs CoT comparison fields: `belief_beats_cot`, `belief_beats_cot_ti`, TI deltas.

## TruthfulQA grading (v7.4)

See [docs/TRUTHFULQA_EVAL.md](../../docs/TRUTHFULQA_EVAL.md). Reports **T**, **I**, **TI**, **misc** on `--truthfulqa` runs.

## Run (mock LLM — CI-safe)

```bash
py -m eval.eidos_eval.runner
py experiments/exp_23_eidos_eval/run.py
py experiments/exp_25_truthfulqa_misconceptions/run.py
```

## Live API eval (Groq — v7.3)

```bash
set GROQ_API_KEY=gsk_...
py -m eval.eidos_eval.live_runner --provider groq
py -m eval.eidos_eval.live_runner --provider groq --truthfulqa
py experiments/exp_24_groq_live_eval/run.py
```

Pilot (N=6): [docs/LIVE_EVAL_PILOT.md](../../docs/LIVE_EVAL_PILOT.md)

Uses **SBERT** + mode-specific gate profiles (`gate_profiles.py`). Reports **task_accuracy**. Responses cached in `live_cache.json` (use `--no-cache` to bypass).

If HuggingFace Hub fails (SBERT download), the runner **falls back to hash** automatically, or force hash:

```bash
py -m eval.eidos_eval.live_runner --provider groq --embedding hash
```

## Gate calibration

```bash
py -m eval.eidos_eval.calibrate_gate
```

Writes `gate_calibration.json` from ambiguous QA benchmark grid search.

## Question sets

| File | N | Use |
|------|---|-----|
| `questions.json` | 8 | Mock CI |
| `questions_live.json` | 6 | Groq pilot |
| `questions_truthfulqa_50.json` | 50 | TruthfulQA Misconceptions live scale |

Build TruthfulQA subset:

```bash
py -m eval.eidos_eval.build_truthfulqa_subset --n 50
```

See `docs/POSITIONING.md` for Core vs Sidecar framing and `docs/SIDECAR_RESEARCH_NOTE.md` for the research arc.

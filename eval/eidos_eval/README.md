# EIDOS-Eval (v7.0)

External comparison harness: **LLM-alone** vs **EIDOS gate** vs **EIDOS meta-injection**.

## Modes

| Mode | `enable_gate` | `enable_meta_injection` |
|------|---------------|-------------------------|
| `llm_alone` | off | off |
| `eidos_gate` | on | off |
| `eidos_meta` | on | on (revision loop) |

## Metrics

- **accuracy** — fraction of questions with correct committed response
- **accuracy_when_commit** — correctness on committed answers only
- **abstention_rate** — fraction withheld (gated non-commit)
- **false_commit_rate** — wrong answer committed on `must_abstain` items
- **must_abstain_safe_rate** — no false commits on abstain-labeled items

## Run (mock LLM — CI-safe)

```bash
py -m eval.eidos_eval.runner
py experiments/exp_23_eidos_eval/run.py
```

## Live API eval (Groq — v7.2)

```bash
set GROQ_API_KEY=gsk_...
py -m eval.eidos_eval.live_runner --provider groq
py experiments/exp_24_groq_live_eval/run.py
```

Uses **SBERT** + calibrated gate (`GATE_LIVE_MIN_DRAFT_GOAL_ALIGN=0.72`). Reports **task_accuracy** (credits correct abstentions). Responses cached in `live_cache.json` (use `--no-cache` to bypass).

Modes: `llm_alone`, `eidos_gate`, `eidos_belief`, `eidos_meta`.

OpenAI still supported via `--provider openai` and `OPENAI_API_KEY`.

## Mock eval (CI-safe)

```bash
py -m eval.eidos_eval.runner
py experiments/exp_23_eidos_eval/run.py
```

## Questions

- `questions.json` — 8 graded items with mock `initial_draft` / `revised_draft` (CI)
- `questions_live.json` — 6 live API items (TruthfulQA-inspired + domains)

See `docs/POSITIONING.md` for Core vs Sidecar framing.

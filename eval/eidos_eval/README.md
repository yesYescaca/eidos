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

## Live API eval (optional)

```bash
set OPENAI_API_KEY=sk-...
set EIDOS_LLM_MODEL=gpt-4o-mini
# Wire OpenAICompatibleLLM in runner via custom hybrid_factory
```

## Questions

`questions.json` — 8 graded items derived from benchmark domains with `initial_draft`, `revised_draft`, and `correct_answer`.

See `docs/POSITIONING.md` for Core vs Sidecar framing.

# EIDOS-Eval (v7.7)

External comparison harness: **LLM-alone** vs **chain-of-thought** vs **self-reflection** vs **EIDOS gate** vs **EIDOS belief** vs **EIDOS meta-injection**.

## Modes

| Mode | Gate | Belief context | Meta revision | CoT | Reflection |
|------|------|----------------|---------------|-----|------------|
| `llm_alone` | off | off | off | off | off |
| `llm_cot` | off | off | off | on | off |
| `llm_reflection` | off | off | off | off | on (2-call) |
| `eidos_gate` | on (calibrated) | off | off | off | off |
| `eidos_belief` | on | on | off | off | off |
| `eidos_meta` | on | on | on | off | off |

## Metrics

- **task_accuracy** — correct committed answers + correct abstentions on `must_abstain` items
- **accuracy** — fraction of questions with correct committed response
- **accuracy_when_commit** — correctness on committed answers only
- **abstention_rate** — fraction withheld (gated non-commit)
- **false_commit_rate** — wrong answer committed on `must_abstain` items
- **must_abstain_safe_rate** — no false commits on abstain-labeled items

Belief vs baseline comparison fields: `belief_beats_cot`, `belief_beats_reflection`, TI and misconception-commit deltas.

## TruthfulQA grading (v7.4)

See [docs/TRUTHFULQA_EVAL.md](../../docs/TRUTHFULQA_EVAL.md). Reports **T**, **I**, **TI**, **misc** on `--truthfulqa` runs.

## Run (mock LLM — CI-safe)

```bash
py -m eval.eidos_eval.runner
py experiments/exp_23_eidos_eval/run.py
py experiments/exp_25_truthfulqa_misconceptions/run.py
```

## Live API eval (Groq — v7.7)

**Important:** Run from the **eidos repository root** (the folder that contains `eval/`, `architecture/`, and `run_live_eval.py`).  
If you see `ModuleNotFoundError: No module named 'eval'`, you are in the wrong directory or wrong virtualenv (e.g. Project AION).

```bash
cd "C:\path\to\eidos"
set GROQ_API_KEY=gsk_...
py run_live_eval.py --provider groq --mixed --modes llm_alone llm_cot llm_reflection eidos_belief
```

Or use the PowerShell wrapper (always uses the eidos repo):

```powershell
& "C:\path\to\eidos\run_live_eval.ps1" --provider groq --mixed --modes llm_alone llm_cot llm_reflection eidos_belief
```

Equivalent module form (only when cwd is eidos root):

```bash
set GROQ_API_KEY=gsk_...
py -m eval.eidos_eval.live_runner --provider groq --mixed --modes llm_alone llm_cot llm_reflection eidos_belief
py -m eval.eidos_eval.live_runner --provider groq --truthfulqa
py -m eval.eidos_eval.live_runner --provider groq --mixed
py -m eval.eidos_eval.live_runner --provider groq --model llama-3.1-8b-instant --truthfulqa
py -m eval.eidos_eval.run_multimodel_eval --provider groq
py experiments/exp_24_groq_live_eval/run.py
```

Pilot (N=6): [docs/LIVE_EVAL_PILOT.md](../../docs/LIVE_EVAL_PILOT.md) · Multi-model: [docs/MULTIMODEL_EVAL.md](../../docs/MULTIMODEL_EVAL.md)

Uses **SBERT** + mode-specific gate profiles (`gate_profiles.py`). Reports **task_accuracy**. Responses cached per model in `live_cache_{model}.json` (legacy `live_cache.json` when `--model` omitted). Use `--no-cache` to bypass.

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

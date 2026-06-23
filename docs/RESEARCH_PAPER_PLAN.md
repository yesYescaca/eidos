# Research Paper — Structure Plan

Full draft: **[RESEARCH_PAPER.md](RESEARCH_PAPER.md)**

## Target venue

Workshop paper or arXiv preprint (cs.AI / cs.CL). Not a Nature/NeurIPS SOTA claim — a **mechanism + eval** contribution from an undergraduate/lab setting.

## Section map

| Section | Purpose | Status |
|---------|---------|--------|
| Abstract | Problem, method, 3-model results, honest caveat | Drafted |
| 1. Introduction | Motivation, dual-process, contributions | Drafted |
| 2. Related Work | TruthfulQA, CoT, calibration, cognitive science | Drafted |
| 3. Method | PAW, hybrid pipeline, eval modes, benchmarks | Drafted |
| 4. Results | 6 tables (3× TruthfulQA, 3× mixed) + summaries | Drafted |
| 5. Discussion | Why mixed > TruthfulQA-only, model effects, limits | Drafted |
| 6. Conclusion | Claims + future work | Drafted |
| References | 8 core citations | Drafted |
| Appendices | Repro, figures, data | Drafted |

## Headline claims (defensible)

1. **Mixed benchmark:** EIDOS belief +26 to +50 task accuracy vs LLM-alone on all 3 Groq models.
2. **Ambiguity:** 88–100% ambig-safe (belief) vs 8–40% (alone).
3. **vs CoT on misconceptions (70B, OSS-20B):** belief commit TI +25 to +35 pts.
4. **Limitation (8B):** CoT beats belief on misconception commits; EIDOS still wins mixed task.

## Before submission checklist

- [ ] Add your full name and email
- [ ] Run `--no-cache` ablation on 70B for robustness paragraph
- [ ] Add Wilson/binomial CIs for N=50 (optional but strengthens paper)
- [ ] Convert to LaTeX (arXiv template)
- [ ] Add pipeline diagram figure
- [ ] Add bar charts from Tables 4–6
- [ ] Peer review from supervisor / Claude pass
- [ ] Decide: lead with mixed (recommended) or TruthfulQA

## Files

| File | Role |
|------|------|
| `docs/RESEARCH_PAPER.md` | Main manuscript draft |
| `docs/LIVE_EVAL_PILOT.md` | Raw result history + v7.5 tables |
| `eval/eidos_eval/reports/` | Machine-readable JSON (local) |
| `docs/SIDECAR_RESEARCH_NOTE.md` | Short arc → points to full paper |

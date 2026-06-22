# Hybrid QA Demo

LLM + EIDOS spike: GPT-2 (or mock) generates; EIDOS gates the response.

```bash
# Mock LLM (no extra deps)
py demos/hybrid_qa/run.py

# Real GPT-2 on CPU (~500MB download first time)
pip install -r requirements-hybrid.txt
py demos/hybrid_qa/run.py --gpt2

# LLM only (no EIDOS gate)
py demos/hybrid_qa/run.py --no-gate
```

See `docs/HYBRID_SPIKE_PLAN.md` for architecture and v6 roadmap.

"""Experiment 28: Multi-model live eval registry (v7.6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.eidos_eval.live_models import GROQ_EVAL_MODELS, model_slug, report_basename
from eval.eidos_eval.llm_cache import cache_path_for_model

OUT = Path(__file__).resolve().parent


def main() -> int:
    models = list(GROQ_EVAL_MODELS)
    slugs = [model_slug(m) for m in models]
    caches = [cache_path_for_model(m).name for m in models]
    unique_slugs = len(set(slugs)) == len(slugs)
    unique_caches = len(set(caches)) == len(caches)
    passed = (
        len(models) >= 2
        and unique_slugs
        and unique_caches
        and report_basename("mixed", models[0]).startswith("live_mixed_")
    )
    payload = {
        "pass": passed,
        "models": models,
        "slugs": slugs,
        "cache_files": caches,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print("EXPERIMENT 28: Multi-model eval registry (v7.6)")
    print(f"  models={len(models)} unique_slugs={unique_slugs}")
    print(f"PASS: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

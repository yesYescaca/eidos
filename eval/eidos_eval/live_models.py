"""Live eval model registry and path helpers (v7.6)."""

from __future__ import annotations

import re
from typing import Literal

from architecture.hybrid.llm_factory import GROQ_DEFAULT_MODEL

LiveEvalProvider = Literal["groq", "openai"]

# Groq models for cross-scale robustness checks (v7.6).
GROQ_EVAL_MODELS: tuple[str, ...] = (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
)

OPENAI_EVAL_MODELS: tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4o",
)

DEFAULT_EVAL_MODELS: dict[LiveEvalProvider, tuple[str, ...]] = {
    "groq": GROQ_EVAL_MODELS,
    "openai": OPENAI_EVAL_MODELS,
}


def default_models_for_provider(provider: LiveEvalProvider) -> tuple[str, ...]:
    return DEFAULT_EVAL_MODELS.get(provider, (GROQ_DEFAULT_MODEL,))


def resolve_model_id(provider: LiveEvalProvider, model: str | None) -> str:
    """Return explicit model or provider default."""
    if model:
        return model
    if provider == "groq":
        import os

        return os.environ.get("GROQ_MODEL", GROQ_DEFAULT_MODEL)
    import os

    return os.environ.get("EIDOS_LLM_MODEL", "gpt-4o-mini")


def model_slug(model_id: str) -> str:
    """Filesystem-safe slug from a model ID."""
    slug = model_id.strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "_", slug)
    return slug.strip("_") or "unknown"


def report_basename(benchmark: str, model_id: str) -> str:
    return f"live_{benchmark}_{model_slug(model_id)}_report.json"

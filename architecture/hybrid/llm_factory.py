"""Live LLM factory — Groq (default) and OpenAI-compatible providers (v7.1)."""

from __future__ import annotations

import os
from typing import Literal

from architecture.hybrid.llm_backend import LanguageModelBackend, OpenAICompatibleLLM

LiveLLMProvider = Literal["groq", "openai"]

GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


def create_live_llm(
    provider: LiveLLMProvider = "groq",
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> LanguageModelBackend:
    """
    Create a live chat-completions LLM backend.

    Groq (default): set GROQ_API_KEY, optional GROQ_MODEL / GROQ_BASE_URL.
    OpenAI: set OPENAI_API_KEY, optional EIDOS_LLM_MODEL / OPENAI_BASE_URL.
    """
    if provider == "groq":
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError(
                "Groq live eval requires GROQ_API_KEY. "
                "Get a key at https://console.groq.com"
            )
        return OpenAICompatibleLLM(
            model=model or os.environ.get("GROQ_MODEL", GROQ_DEFAULT_MODEL),
            api_key=key,
            base_url=base_url or os.environ.get("GROQ_BASE_URL", GROQ_DEFAULT_BASE_URL),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OpenAI live eval requires OPENAI_API_KEY")
        return OpenAICompatibleLLM(
            model=model or os.environ.get("EIDOS_LLM_MODEL", OPENAI_DEFAULT_MODEL),
            api_key=key,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", OPENAI_DEFAULT_BASE_URL),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    raise ValueError(f"Unknown live LLM provider: {provider}")


def live_llm_available(provider: LiveLLMProvider = "groq") -> bool:
    """Return True if API key for provider is set in environment."""
    if provider == "groq":
        return bool(os.environ.get("GROQ_API_KEY"))
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    return False

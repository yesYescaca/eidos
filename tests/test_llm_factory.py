"""Tests for v7.1 live LLM factory."""

import os

import pytest

from architecture.hybrid.llm_factory import create_live_llm, live_llm_available


def test_live_llm_available_false_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert live_llm_available("groq") is False


def test_create_live_llm_groq_requires_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        create_live_llm("groq")


def test_create_live_llm_groq_with_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    llm = create_live_llm("groq", model="llama-3.3-70b-versatile")
    assert llm.model == "llama-3.3-70b-versatile"
    assert llm.api_key == "test-key"
    assert "groq.com" in llm.base_url


def test_create_live_grounding_hash_fallback():
    from architecture.bridge.embedding_factory import (
        create_live_grounding,
        get_hash_grounding,
        reset_grounding_cache,
        resolve_active_backend,
    )

    reset_grounding_cache()
    g = create_live_grounding(prefer_sbert=False)
    assert resolve_active_backend(g) == "hash"
    assert g is get_hash_grounding()
    reset_grounding_cache()


def test_openai_compatible_llm_includes_user_agent():
    from architecture.hybrid.llm_backend import OpenAICompatibleLLM

    headers = OpenAICompatibleLLM._request_headers("test-key")
    assert "User-Agent" in headers
    assert "EIDOS-LiveEval/7.2" in headers["User-Agent"]
    assert headers["Accept"] == "application/json"
    assert headers["Authorization"] == "Bearer test-key"

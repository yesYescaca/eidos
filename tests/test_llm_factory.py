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

"""Hybrid LLM + EIDOS — optional language model backends."""

from architecture.hybrid.llm_backend import GPT2LanguageModel, MockLanguageModel

__all__ = ["GPT2LanguageModel", "MockLanguageModel"]

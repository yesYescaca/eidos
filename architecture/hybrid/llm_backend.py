"""LLM backends for hybrid spike — mock (tests) and GPT-2 (demo)."""

from __future__ import annotations

from typing import Protocol


class LanguageModelBackend(Protocol):
  def generate(self, prompt: str, max_new_tokens: int = 48) -> str:
    ...


class MockLanguageModel:
    """
    Deterministic stub LLM for experiments and CI.

    Returns a draft biased toward 'beta' when prompt mentions sector,
    simulating a confidently wrong small model.
    """

    def __init__(self, bias: str = "beta", draft: str | None = None) -> None:
        self.bias = bias
        self._draft_override = draft

    def generate(self, prompt: str, max_new_tokens: int = 48) -> str:
        if self._draft_override is not None:
            return self._draft_override
        if self.bias == "beta":
            return (
                "The reactor core is overheating in sector beta. "
                "Recommend immediate cooling protocol."
            )
        return (
            "The reactor core is overheating in sector alpha. "
            "Recommend immediate cooling protocol."
        )


class CaseMockLLM:
    """Alias used by benchmark — configured draft per case."""

    def __init__(self, draft: str) -> None:
        self.draft = draft

    def generate(self, prompt: str, max_new_tokens: int = 48) -> str:
        return self.draft


class GPT2LanguageModel:
    """Small GPT-2 text generation on CPU (optional — requires transformers)."""

    def __init__(self, model_name: str = "gpt2", max_new_tokens: int = 48) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "GPT2LanguageModel requires transformers. "
                "Install: pip install -r requirements-hybrid.txt"
            ) from exc

        self._max_new_tokens = max_new_tokens
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            device=-1,
        )

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        tokens = max_new_tokens if max_new_tokens is not None else self._max_new_tokens
        outputs = self._pipe(
            prompt,
            max_new_tokens=tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=50256,
        )
        text = outputs[0]["generated_text"]
        if text.startswith(prompt):
            return text[len(prompt) :].strip()
        return text.strip()

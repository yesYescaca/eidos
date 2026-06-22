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


class RoundRobinMockLLM:
    """
    Returns drafts in sequence — simulates LLM revising after meta-injection.

    First call: wrong draft; subsequent calls: revised drafts from list.
    """

    def __init__(self, drafts: list[str]) -> None:
        if not drafts:
            raise ValueError("RoundRobinMockLLM requires at least one draft")
        self._drafts = list(drafts)
        self._index = 0

    def generate(self, prompt: str, max_new_tokens: int = 48) -> str:
        draft = self._drafts[min(self._index, len(self._drafts) - 1)]
        self._index += 1
        return draft


class OpenAICompatibleLLM:
    """
    OpenAI-compatible chat API (optional — requires OPENAI_API_KEY).

    Also works with local servers (Ollama, vLLM) via OPENAI_BASE_URL.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        import os

        self.model = model or os.environ.get("EIDOS_LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        if not self.api_key:
            raise ValueError(
                "OpenAICompatibleLLM requires OPENAI_API_KEY or explicit api_key"
            )

    @staticmethod
    def _request_headers(api_key: str) -> dict[str, str]:
        """Headers for chat-completions (Groq/Cloudflare rejects default urllib UA)."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "EIDOS-LiveEval/7.2 (compatible; +https://github.com/yesYescaca/eidos)",
        }

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        import json
        import time
        import urllib.error
        import urllib.request

        from agent.config import GATE_LLM_RETRY_ATTEMPTS, GATE_LLM_RETRY_BACKOFF_SEC

        tokens = max_new_tokens if max_new_tokens is not None else self.max_tokens
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": tokens,
                "temperature": self.temperature,
            }
        ).encode("utf-8")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = self._request_headers(self.api_key)
        last_error: Exception | None = None

        for attempt in range(max(1, GATE_LLM_RETRY_ATTEMPTS)):
            req = urllib.request.Request(
                url,
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(f"LLM API returned no choices: {data}")
                message = choices[0].get("message", {})
                return str(message.get("content", "")).strip()
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"LLM API error {exc.code}: {body}")
                if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < GATE_LLM_RETRY_ATTEMPTS:
                    time.sleep(GATE_LLM_RETRY_BACKOFF_SEC * (attempt + 1))
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"LLM API connection error: {exc}")
                if attempt + 1 < GATE_LLM_RETRY_ATTEMPTS:
                    time.sleep(GATE_LLM_RETRY_BACKOFF_SEC * (attempt + 1))
                    continue
                raise last_error from exc

        if last_error:
            raise last_error
        raise RuntimeError("LLM API request failed")


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

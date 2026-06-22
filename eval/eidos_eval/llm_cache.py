"""Disk cache for live LLM responses (v7.2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from architecture.hybrid.llm_backend import LanguageModelBackend

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "live_cache.json"


class CachedLLM:
    """Wrap an LLM backend with a JSON file cache keyed by model + prompt."""

    def __init__(
        self,
        backend: LanguageModelBackend,
        cache_path: Path | None = None,
    ) -> None:
        self._backend = backend
        self._path = cache_path or DEFAULT_CACHE_PATH
        self._cache: dict[str, str] = {}
        if self._path.exists():
            try:
                self._cache = dict(json.loads(self._path.read_text()))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    @property
    def model(self) -> str:
        return str(getattr(self._backend, "model", "unknown"))

    def _key(self, prompt: str) -> str:
        raw = f"{self.model}\n{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        key = self._key(prompt)
        if key in self._cache:
            return self._cache[key]
        text = self._backend.generate(prompt, max_new_tokens=max_new_tokens)
        self._cache[key] = text
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._cache, indent=2))
        return text

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self._cache), "path": str(self._path)}

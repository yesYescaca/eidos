"""Text grounding bridge — phrases to concept vectors (v5.0)."""

from __future__ import annotations

import re

import numpy as np

from agent.config import INPUT_DIM, TEXT_GROUNDING_BUCKETS, TEXT_GROUNDING_SEED


class TextGroundingBridge:
    """
    Deterministic text → vector embedding via hashed n-grams + random projection.

    Inspired by distributional semantics (Landauer & Dumais 1997) and symbol
    grounding (Harnad 1990). No ML frameworks — numpy only, laptop-friendly.
    """

    def __init__(
        self,
        dim: int = INPUT_DIM,
        seed: int = TEXT_GROUNDING_SEED,
        n_buckets: int = TEXT_GROUNDING_BUCKETS,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self.n_buckets = n_buckets
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(n_buckets)
        self._projection = rng.normal(0, scale, (dim, n_buckets))

    def _bucket(self, token: str) -> int:
        return hash((token, self.seed)) % self.n_buckets

    def _normalise(self, text: str) -> str:
        text = text.lower().strip()
        return re.sub(r"[^a-z0-9\s]", "", text)

    def embed(self, text: str) -> np.ndarray:
        """Map a phrase to a unit-norm vector in R^dim."""
        buckets = np.zeros(self.n_buckets, dtype=np.float64)
        normalised = self._normalise(text)

        for word in normalised.split():
            if word:
                buckets[self._bucket(f"w:{word}")] += 2.0

        compact = normalised.replace(" ", "")
        for i in range(max(0, len(compact) - 2)):
            trigram = compact[i : i + 3]
            buckets[self._bucket(f"t:{trigram}")] += 1.0

        vec = self._projection @ buckets
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        return vec.astype(np.float64)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two embedded phrases."""
        a = self.embed(text_a)
        b = self.embed(text_b)
        return float(np.dot(a, b))

"""Optional sentence-transformer embeddings (CPU)."""

from __future__ import annotations

import numpy as np

from agent.config import INPUT_DIM, SBERT_MODEL_NAME, TEXT_GROUNDING_SEED


class SentenceTransformerGrounding:
    """
    Semantic embeddings via sentence-transformers, projected to PAW input_dim.

    Requires: pip install -r requirements-embeddings.txt
    """

    def __init__(
        self,
        dim: int = INPUT_DIM,
        model_name: str = SBERT_MODEL_NAME,
        seed: int = TEXT_GROUNDING_SEED,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerGrounding requires sentence-transformers. "
                "Install: pip install -r requirements-embeddings.txt"
            ) from exc

        self.dim = dim
        self.model_name = model_name
        self._model = self._load_model(model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self._model.get_sentence_embedding_dimension
        raw_dim = get_dim()
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(raw_dim)
        self._projection = rng.normal(0, scale, (dim, raw_dim))

    def embed(self, text: str) -> np.ndarray:
        raw = np.asarray(
            self._model.encode(text, normalize_embeddings=True),
            dtype=np.float64,
        )
        if raw.shape[0] == self.dim:
            return raw
        vec = self._projection @ raw
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        return vec.astype(np.float64)

    def similarity(self, text_a: str, text_b: str) -> float:
        a = self.embed(text_a)
        b = self.embed(text_b)
        return float(np.dot(a, b))

    @staticmethod
    def _load_model(model_name: str):
        """Load SBERT, preferring local cache when Hub is flaky."""
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            return SentenceTransformer(model_name)

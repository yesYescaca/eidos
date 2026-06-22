"""Text embedding backend factory (v6.0)."""

from __future__ import annotations

from typing import Literal

from agent.config import HYBRID_DEFAULT_EMBEDDING

from architecture.bridge.sentence_transformer_grounding import SentenceTransformerGrounding
from architecture.bridge.text_grounding import TextGroundingBridge

EmbeddingBackend = Literal["hash", "sbert"]


def create_grounding(
    backend: EmbeddingBackend = "hash",
    **kwargs,
) -> TextGroundingBridge | SentenceTransformerGrounding:
    """Create a text grounding backend."""
    if backend == "hash":
        return TextGroundingBridge(**kwargs)
    if backend == "sbert":
        return SentenceTransformerGrounding(**kwargs)
    raise ValueError(f"Unknown embedding backend: {backend}")


def create_hybrid_grounding(**kwargs) -> TextGroundingBridge | SentenceTransformerGrounding:
    """Hybrid path default: SBERT if available, else hash (CI-safe)."""
    preferred = HYBRID_DEFAULT_EMBEDDING
    if preferred == "sbert":
        try:
            return create_grounding("sbert", **kwargs)
        except ImportError:
            pass
    return create_grounding("hash", **kwargs)


def resolve_hybrid_embedding_backend() -> EmbeddingBackend:
    """Return backend name that create_hybrid_grounding would use."""
    if HYBRID_DEFAULT_EMBEDDING == "sbert":
        try:
            import sentence_transformers  # noqa: F401
            return "sbert"
        except ImportError:
            return "hash"
    return "hash"

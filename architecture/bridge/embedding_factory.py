"""Text embedding backend factory (v6.0+)."""

from __future__ import annotations

import warnings
from typing import Literal

from agent.config import HYBRID_DEFAULT_EMBEDDING

from architecture.bridge.text_grounding import TextGroundingBridge

EmbeddingBackend = Literal["hash", "sbert"]

# Process-wide singleton — avoid re-downloading SBERT for every HybridEidosAgent.
_SBERT_GROUNDING: object | None = None
_HASH_GROUNDING: TextGroundingBridge | None = None


def create_grounding(
    backend: EmbeddingBackend = "hash",
    **kwargs,
):
    """Create a text grounding backend."""
    if backend == "hash":
        return TextGroundingBridge(**kwargs)
    if backend == "sbert":
        from architecture.bridge.sentence_transformer_grounding import (
            SentenceTransformerGrounding,
        )

        return SentenceTransformerGrounding(**kwargs)
    raise ValueError(f"Unknown embedding backend: {backend}")


def get_sbert_grounding(**kwargs):
    """Return cached SBERT grounding, loading once per process."""
    global _SBERT_GROUNDING
    if _SBERT_GROUNDING is not None:
        return _SBERT_GROUNDING
    _SBERT_GROUNDING = create_grounding("sbert", **kwargs)
    return _SBERT_GROUNDING


def get_hash_grounding(**kwargs) -> TextGroundingBridge:
    """Return cached hash grounding."""
    global _HASH_GROUNDING
    if _HASH_GROUNDING is None:
        _HASH_GROUNDING = create_grounding("hash", **kwargs)
    return _HASH_GROUNDING


def reset_grounding_cache() -> None:
    """Clear singleton cache (tests only)."""
    global _SBERT_GROUNDING, _HASH_GROUNDING
    _SBERT_GROUNDING = None
    _HASH_GROUNDING = None


def create_hybrid_grounding(**kwargs):
    """Hybrid path default: SBERT if available, else hash (CI-safe)."""
    preferred = HYBRID_DEFAULT_EMBEDDING
    if preferred == "sbert":
        try:
            return get_sbert_grounding(**kwargs)
        except ImportError:
            pass
        except Exception as exc:
            warnings.warn(
                f"SBERT load failed ({exc!r}); falling back to hash embeddings.",
                stacklevel=2,
            )
    return get_hash_grounding(**kwargs)


def create_live_grounding(
    *,
    prefer_sbert: bool = True,
    **kwargs,
):
    """
    Grounding for live API eval — singleton SBERT with hash fallback.

    Use when HuggingFace Hub is unreachable or SBERT is not installed.
    """
    if prefer_sbert:
        try:
            return get_sbert_grounding(**kwargs)
        except ImportError:
            warnings.warn(
                "sentence-transformers not installed; live eval using hash embeddings.",
                stacklevel=2,
            )
        except Exception as exc:
            warnings.warn(
                f"SBERT unavailable for live eval ({exc!r}); using hash embeddings.",
                stacklevel=2,
            )
    return get_hash_grounding(**kwargs)


def resolve_hybrid_embedding_backend() -> EmbeddingBackend:
    """Return backend name that create_hybrid_grounding would use."""
    if HYBRID_DEFAULT_EMBEDDING == "sbert":
        try:
            import sentence_transformers  # noqa: F401

            return "sbert"
        except ImportError:
            return "hash"
    return "hash"


def resolve_active_backend(grounding: object) -> EmbeddingBackend:
    """Return backend label for an existing grounding instance."""
    name = type(grounding).__name__
    if "SentenceTransformer" in name:
        return "sbert"
    return "hash"

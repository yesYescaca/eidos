"""Text embedding backend factory (v6.0)."""

from __future__ import annotations

from typing import Literal

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

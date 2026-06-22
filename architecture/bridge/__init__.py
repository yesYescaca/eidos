"""Text bridge package."""

from architecture.bridge.embedding_factory import (
    create_grounding,
    create_hybrid_grounding,
    resolve_hybrid_embedding_backend,
)
from architecture.bridge.text_grounding import TextGroundingBridge

__all__ = [
    "TextGroundingBridge",
    "create_grounding",
    "create_hybrid_grounding",
    "resolve_hybrid_embedding_backend",
]

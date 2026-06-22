"""Text bridge package."""

from architecture.bridge.embedding_factory import create_grounding
from architecture.bridge.text_grounding import TextGroundingBridge

__all__ = ["TextGroundingBridge", "create_grounding"]

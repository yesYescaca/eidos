"""Detect genuine underdetermination in user questions (v7.5)."""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

UNDERDETERMINATION_MARKERS = (
    "unknown",
    "unclear",
    "unsure",
    "uncertain",
    "ambiguous",
    "could be",
    "might be",
    "not sure",
    "either",
    " or ",
    "which one",
    "which scenario",
    "unsure if",
    "unclear if",
    "unclear whether",
    "sector unknown",
    "unclear source",
    "unclear primary",
    "unclear if hacked",
    "unsure if",
    "unsure fraud",
    "unclear stroke",
    "unclear if voluntary",
    "unclear breach",
    "unsure real fire",
    "unclear plagiarism",
    "unclear out of stock",
)


def has_underdetermination_markers(text: str) -> bool:
    """True when the question signals genuine ambiguity, not just unfamiliar phrasing."""
    norm = _normalize(text)
    if not norm:
        return False
    markers = [_normalize(m) for m in UNDERDETERMINATION_MARKERS]
    return any(marker in norm for marker in markers)

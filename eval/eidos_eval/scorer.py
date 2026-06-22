"""Answer scoring for EIDOS-Eval (v7.0)."""

from __future__ import annotations

import re


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_correct(response: str, correct_answer: str) -> bool:
    """Check if response contains the expected answer phrase."""
    if not response or not correct_answer:
        return False
    return normalize(correct_answer) in normalize(response)


def committed(response: str, gate_decision: str, gated: bool) -> bool:
    """Whether the system delivered an LLM answer (not abstention message)."""
    if gate_decision == "commit":
        return True
    return not gated

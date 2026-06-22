"""Answer scoring for EIDOS-Eval (v7.0)."""

from __future__ import annotations

import re


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_correct(
    response: str,
    correct_answer: str,
    *,
    gate_decision: str = "",
    gated: bool = False,
) -> bool:
    """Check if response matches expected outcome (substring or abstention)."""
    if not correct_answer:
        return False
    if normalize(correct_answer) == "clarify":
        if gated or gate_decision in ("clarify", "defer", "probe", "sleep"):
            return True
        markers = (
            "clarify",
            "which scenario",
            "more than one",
            "not confident",
            "unclear",
            "cannot determine",
            "need more",
        )
        norm = normalize(response)
        return any(marker in norm for marker in markers)
    if not response:
        return False
    return normalize(correct_answer) in normalize(response)


def committed(response: str, gate_decision: str, gated: bool) -> bool:
    """Whether the system delivered an LLM answer (not abstention message)."""
    if gate_decision == "commit":
        return True
    return not gated


def task_handled_correctly(
    *,
    must_abstain: bool,
    is_committed: bool,
    false_commit: bool,
    answer_ok: bool,
) -> bool:
    """
    Whether the system took the right action for the question type.

    Ambiguous (must_abstain): withhold or clarify — not a wrong commit.
    Clear: commit with a correct answer.
    """
    if must_abstain:
        return not false_commit
    return is_committed and answer_ok


def selective_accuracy_delta(
    alone_commit_acc: float,
    sidecar_commit_acc: float,
) -> float:
    """Accuracy-on-commits delta: Sidecar vs LLM-alone."""
    return sidecar_commit_acc - alone_commit_acc

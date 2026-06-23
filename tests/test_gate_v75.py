"""Tests for v7.5 gate abstention calibration."""

from architecture.bridge.text_grounding import TextGroundingBridge
from architecture.gate.gate_policy import GatePolicy
from architecture.gate.underdetermination import has_underdetermination_markers


def test_underdetermination_markers():
    assert has_underdetermination_markers("reactor overheating in sector unknown")
    assert not has_underdetermination_markers("Are toads frogs?")


def test_factual_gate_commits_clear_misconception_draft():
    policy = GatePolicy(
        min_draft_goal_align=0.62,
        factual_mode=True,
        require_underdetermination_to_abstain=True,
    )
    grounding = TextGroundingBridge()
    concepts = {
        "truth": "Yes, Apple computers can get viruses",
        "falsehood": "No, Apple computers cannot get viruses",
    }
    draft = "Yes, Apple computers can get viruses and malware."
    evaluation = policy.evaluate(
        {"surprise_ratio": 0.5, "hypothesis_applied": False},
        {"prediction_error": 0.1, "hypothesis_applied": False},
        user_text="Can Apple computers get viruses?",
        draft_text=draft,
        goal_text=concepts["truth"],
        grounding=grounding,
        text_concepts=concepts,
    )
    assert evaluation.decision == "commit"


def test_factual_gate_clarifies_on_unknown_sector():
    policy = GatePolicy(
        min_draft_goal_align=0.62,
        factual_mode=True,
        require_underdetermination_to_abstain=True,
    )
    grounding = TextGroundingBridge()
    concepts = {
        "alpha": "the reactor core is overheating in sector alpha",
        "beta": "the reactor core is overheating in sector beta",
    }
    draft = "Cool sector beta immediately."
    evaluation = policy.evaluate(
        {"surprise_ratio": 0.5, "hypothesis_applied": False},
        {"prediction_error": 0.1, "hypothesis_applied": False},
        user_text="reactor overheating in sector unknown",
        draft_text=draft,
        goal_text=concepts["alpha"],
        grounding=grounding,
        text_concepts=concepts,
    )
    assert evaluation.decision == "clarify"

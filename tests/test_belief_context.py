"""Tests for v7.1 belief-grounded prompts."""

from architecture.bridge.text_grounding import TextGroundingBridge
from architecture.hybrid.belief_context import (
    build_belief_context,
    build_grounded_prompt,
    rank_concepts_for_text,
)


def test_rank_concepts_orders_by_similarity():
    grounding = TextGroundingBridge()
    concepts = {
        "fire": "smoke and flames spreading through the building",
        "water": "cold water flooding the basement floor",
    }
    ranked = rank_concepts_for_text(
        "smoke in the hallway",
        concepts,
        grounding,
    )
    assert len(ranked) == 2
    assert ranked[0][1] >= ranked[1][1]


def test_build_belief_context_contains_concepts():
    grounding = TextGroundingBridge()
    concepts = {
        "alpha": "the reactor core is overheating in sector alpha",
        "beta": "the reactor core is overheating in sector beta",
    }
    step = {"surprise_ratio": 1.8, "meta_cognition_flags": ["low_confidence"]}
    block = build_belief_context(
        step,
        text_concepts=concepts,
        grounding=grounding,
        user_text="reactor overheating sector unknown",
        goal_text=concepts["alpha"],
    )
    assert "EIDOS Belief State" in block
    assert "Surprise ratio" in block
    assert "alpha" in block.lower()


def test_build_grounded_prompt_prepends_context():
    prompt = build_grounded_prompt(
        "What should we do?",
        "[EIDOS Belief State]\nTop concepts: fire (0.9)",
    )
    assert "Belief State" in prompt
    assert "What should we do?" in prompt

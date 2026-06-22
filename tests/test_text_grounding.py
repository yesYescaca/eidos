"""Tests for v5.0 text grounding bridge."""

import json
import tempfile
from pathlib import Path

import numpy as np

from agent.text_agent import EidosTextAgent, interpret_text_decision
from architecture.bridge.text_grounding import TextGroundingBridge


def test_similar_phrases_embed_closer_than_unrelated():
    bridge = TextGroundingBridge()
    a = "the fire is spreading through the building"
    b = "the fire is growing through the building"
    c = "the ocean is calm and blue today"
    sim_ab = bridge.similarity(a, b)
    sim_ac = bridge.similarity(a, c)
    assert sim_ab > sim_ac


def test_text_agent_register_and_step():
    agent = EidosTextAgent(seed=0, enable_meta_cognition=False)
    agent.register_text_concept("fire", "fire alarm triggered in hallway")
    result = agent.step_text("fire alarm triggered in hallway", input_label="fire")
    assert result["text_decision"] in ("observe", "commit", "probe")
    assert "source_text" in result


def test_defer_decision_mapping():
    decision = interpret_text_decision({
        "meta_cognition_flags": ["hypothesis_deferred"],
        "hypothesis_applied": False,
    })
    assert decision == "defer"


def test_text_agent_save_load_roundtrip():
    agent = EidosTextAgent(seed=1)
    agent.register_text_concept("fire", "flames in the kitchen")
    agent.step_text("flames in the kitchen", "fire")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "text_agent.json"
        agent.save_state(path)
        loaded = json.loads(path.read_text())
        assert loaded["version"] == "7.1"
        assert loaded["text_concepts"]["fire"] == "flames in the kitchen"

        agent2 = EidosTextAgent(seed=99)
        agent2.load_state(path)
        assert agent2._text_concepts["fire"] == "flames in the kitchen"

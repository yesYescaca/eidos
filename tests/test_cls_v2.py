"""Tests for v2.0 BeliefGraph, sleep replay, and CLS recovery inference."""

import numpy as np

from architecture.components.association_store import AssociationStore
from architecture.components.belief_graph import BeliefGraph
from architecture.components.episodic_buffer import EpisodicBuffer
from architecture.components.sleep_replay import SleepReplay
from agent.eidos import EidosAgent


def test_belief_graph_dominant_concept():
    bg = BeliefGraph()
    fire = np.ones(64)
    water = np.ones(64) * 2
    for _ in range(10):
        bg.integrate("fire", fire)
    for _ in range(3):
        bg.integrate("water", water)

    label, strength = bg.dominant_concept()
    assert label == "fire"
    assert strength == 10.0


def test_sleep_replay_strengthens_beliefs():
    rng = np.random.default_rng(0)
    buffer = EpisodicBuffer()
    bg = BeliefGraph()
    fire = rng.normal(0, 1, 64)
    for _ in range(30):
        buffer.record("fire", fire + rng.normal(0, 0.05, 64))

    replay = SleepReplay(seed=0)
    associations = AssociationStore()
    result = replay.run(buffer, bg, associations)

    assert result["replays"] == 30
    assert bg.strength("fire") > 0


def test_v2_misleading_context_uses_belief_graph():
    rng = np.random.default_rng(1)
    agent = EidosAgent(seed=1, enable_reasoning=True, apply_hypotheses=True)
    base = rng.normal(0, 1, 64)
    fire, water = base.copy(), base + rng.normal(0, 2, 64)
    agent.register_concept("fire", fire)
    agent.register_concept("water", water)

    for _ in range(50):
        agent.step(fire + rng.normal(0, 0.05, 64), "fire")
    agent.sleep()

    for _ in range(12):
        agent.step(water + rng.normal(0, 0.05, 64), "water")

    surprise = rng.normal(0, 8, 64)
    result = agent.step(surprise, "anomaly")

    assert result["recovery_inference_source"] == "belief_graph"
    assert result["inferred_recovery_label"] == "fire"


def test_v2_cold_start_uses_belief_graph():
    rng = np.random.default_rng(2)
    agent = EidosAgent(seed=2, enable_reasoning=True, apply_hypotheses=True)
    fire = rng.normal(0, 1, 64)
    agent.register_concept("fire", fire)
    for _ in range(40):
        agent.step(fire + rng.normal(0, 0.05, 64), "fire")
    agent.sleep()

    agent.workspace.clear()
    agent._recovery_context.clear()

    surprise = rng.normal(0, 8, 64)
    result = agent.step(surprise, "anomaly")

    assert result["recovery_inference_source"] == "belief_graph"
    assert result["inferred_recovery_label"] == "fire"

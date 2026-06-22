"""Tests for v3.0 meta-cognition."""

import numpy as np

from architecture.components.episodic_buffer import EpisodicBuffer
from architecture.components.meta_cognition import MetaCognitionMonitor
from agent.eidos import EidosAgent


def test_detect_misleading_context():
    buffer = EpisodicBuffer()
    fire = np.ones(64)
    water = np.ones(64) * 2
    for _ in range(50):
        buffer.record("fire", fire)
    for _ in range(10):
        buffer.record("water", water)

    meta = MetaCognitionMonitor(misleading_context_ratio=2.0)
    is_misleading, dominant, _, _ = meta.detect_misleading_context(
        recent_label="water",
        recent_window_count=10,
        episodic_buffer=buffer,
        registered={"fire", "water"},
    )
    assert is_misleading
    assert dominant == "fire"


def test_agent_meta_prevents_misleading_recovery():
    rng = np.random.default_rng(0)
    agent = EidosAgent(seed=0, enable_meta_cognition=True)
    base = rng.normal(0, 1, 64)
    agent.register_concept("fire", base.copy())
    agent.register_concept("water", base + rng.normal(0, 2, 64))

    for _ in range(50):
        agent.step(base + rng.normal(0, 0.05, 64), "fire")
    for _ in range(12):
        agent.step(base + rng.normal(0, 2, 64) + rng.normal(0, 0.05, 64), "water")

    surprise = rng.normal(0, 8, 64)
    result = agent.step(surprise, "anomaly")

    assert "misleading_context_detected" in result["meta_cognition_flags"]
    assert result["inferred_recovery_label"] == "fire"
    assert result["recovery_inference_source"] == "meta_cognition"


def test_meta_disabled_preserves_recent_inference():
    rng = np.random.default_rng(1)
    agent = EidosAgent(seed=1, enable_meta_cognition=False)
    base = rng.normal(0, 1, 64)
    fire, water = base.copy(), base + rng.normal(0, 2, 64)
    agent.register_concept("fire", fire)
    agent.register_concept("water", water)

    for _ in range(40):
        agent.step(fire + rng.normal(0, 0.05, 64), "fire")
    for _ in range(12):
        agent.step(water + rng.normal(0, 0.05, 64), "water")

    surprise = rng.normal(0, 8, 64)
    result = agent.step(surprise, "anomaly")

    assert result["inferred_recovery_label"] == "water"
    assert result["recovery_inference_source"] == "recent_history"

"""Tests for autonomous recovery probe inference (v1.4)."""

import numpy as np

from architecture.components.recovery_context import RecoveryContextTracker
from agent.eidos import EidosAgent


def test_recovery_context_infers_dominant_label():
    tracker = RecoveryContextTracker(window=10)
    fire = np.ones(64)
    water = np.ones(64) * 2.0
    concepts = {"fire": fire, "water": water}

    for _ in range(7):
        tracker.record("fire", fire + np.random.normal(0, 0.01, 64))
    for _ in range(2):
        tracker.record("water", water)

    probe, label, source = tracker.infer_recovery_probe(
        concepts, "anomaly", input_dim=64
    )
    assert label == "fire"
    assert source == "recent_history"
    assert probe is not None
    assert np.allclose(probe, fire, atol=0.1)


def test_recovery_context_workspace_fallback():
    tracker = RecoveryContextTracker(window=5)
    fire = np.random.randn(64)
    water = np.random.randn(64)
    concepts = {"fire": fire, "water": water}

    probe, label, source = tracker.infer_recovery_probe(
        concepts,
        "anomaly",
        input_dim=64,
        workspace_labels=["fire", "fire", "anomaly"],
    )
    assert label == "fire"
    assert source == "workspace"
    assert np.allclose(probe, fire)


def test_agent_infers_recovery_on_surprise():
    rng = np.random.default_rng(0)
    agent = EidosAgent(seed=0, enable_reasoning=True, apply_hypotheses=True)
    stable = rng.normal(0, 1, 64)
    agent.register_concept("fire", stable)
    agent.register_concept("water", stable + rng.normal(0, 2, 64))

    for _ in range(15):
        vec = stable + rng.normal(0, 0.05, 64)
        agent.step(vec, "fire")

    surprise = rng.normal(0, 6, 64)
    result = agent.step(surprise, "anomaly")

    assert result["recovery_inference_source"] == "recent_history"
    assert result["inferred_recovery_label"] == "fire"
    if result["hypothesis"]:
        assert result["hypothesis"].get("associate") == "fire"


def test_explicit_recovery_probe_overrides_inference():
    rng = np.random.default_rng(1)
    agent = EidosAgent(seed=1)
    fire = rng.normal(0, 1, 64)
    water = fire + rng.normal(0, 2, 64)
    agent.register_concept("fire", fire)
    agent.register_concept("water", water)

    for _ in range(10):
        agent.step(fire + rng.normal(0, 0.05, 64), "fire")

    surprise = rng.normal(0, 6, 64)
    result = agent.step(surprise, "anomaly", recovery_probe=water)

    assert result["recovery_inference_source"] == "explicit"
    assert result["inferred_recovery_label"] is None


def test_save_load_recovery_context():
    import tempfile
    from pathlib import Path

    agent = EidosAgent(seed=2)
    vec = np.ones(64)
    agent.register_concept("fire", vec)
    agent.step(vec, "fire")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        agent.save_state(path)

        loaded = EidosAgent(seed=2)
        loaded.load_state(path)
        assert len(loaded._recovery_context._history) == 1

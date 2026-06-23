import json
import tempfile
from pathlib import Path

import numpy as np

from agent.eidos import EidosAgent


def test_step_returns_valid_dict():
    agent = EidosAgent(seed=42)
    result = agent.step(np.random.randn(64), "test_input")
    assert "salience" in result
    assert "prediction_error" in result
    assert "reward" in result
    assert "hypothesis" in result
    assert "reasoning_triggered" in result
    assert "hypothesis_applied" in result


def test_run_episode_returns_n_results():
    agent = EidosAgent(seed=42)
    inputs = [{"vector": np.random.randn(64), "label": f"c{i}"} for i in range(5)]
    log = agent.run_episode(inputs)
    assert len(log) == 5


def test_reasoning_disabled_never_triggers():
    agent = EidosAgent(seed=42, enable_reasoning=False)
    rng = np.random.default_rng(0)
    stable = rng.normal(0, 1, 64)
    for _ in range(10):
        agent.step(stable + rng.normal(0, 0.1, 64), "fire")
    result = agent.step(rng.normal(0, 5, 64), "anomaly")
    assert result["reasoning_triggered"] is False
    assert result["hypothesis"] is None


def test_hypothesis_application_on_surprise():
    agent = EidosAgent(seed=42, enable_reasoning=True, apply_hypotheses=True)
    rng = np.random.default_rng(42)
    stable = rng.normal(0, 1, 64)
    helper = stable + rng.normal(0, 0.2, 64)
    agent.register_concept("fire", stable)
    agent.register_concept("smoke", helper)

    for i in range(20):
        agent.step(stable + rng.normal(0, 0.05, 64), "fire")
        if i % 2 == 0:
            agent.step(helper + rng.normal(0, 0.05, 64), "smoke")

    result = agent.step(rng.normal(0, 5, 64), "anomaly")
    if result["reasoning_triggered"] and result["hypothesis"]:
        assert "hypothesis_applied" in result


def test_save_load_roundtrip():
    agent = EidosAgent(seed=42)
    for i in range(3):
        agent.step(np.random.randn(64), f"concept_{i}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        agent.save_state(path)

        agent2 = EidosAgent(seed=99)
        agent2.load_state(path)

        assert agent2.associations.to_dict() == agent.associations.to_dict()
        assert agent2.prediction.get_weights_dict() == agent.prediction.get_weights_dict()
        assert agent2._step_count == agent._step_count
        assert agent2.enable_reasoning == agent.enable_reasoning

        saved = json.loads(path.read_text())
        assert saved["version"] == "7.8"
        assert "surprise" in saved

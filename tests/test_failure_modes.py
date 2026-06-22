"""Sanity tests for failure-mode experiment logic."""

import numpy as np

from agent.eidos import EidosAgent


def _train_fire(agent: EidosAgent, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    base = rng.normal(0, 1, 64)
    fire = base.copy()
    water = base + rng.normal(0, 2.5, 64)
    agent.register_concept("fire", fire)
    agent.register_concept("water", water)
    for _ in range(40):
        agent.step(fire + rng.normal(0, 0.05, 64), "fire")
    return fire, water


def test_misleading_context_infers_decoy():
    rng = np.random.default_rng(0)
    agent = EidosAgent(seed=0, enable_reasoning=True, apply_hypotheses=True)
    fire, water = _train_fire(agent, rng)

    for _ in range(12):
        agent.step(water + rng.normal(0, 0.05, 64), "water")

    surprise = rng.normal(0, 8, 64)
    result = agent.step(surprise, "anomaly")

    assert result["inferred_recovery_label"] == "water"
    if result["hypothesis"]:
        assert result["hypothesis"].get("associate") == "water"


def test_cold_start_has_no_history_inference():
    rng = np.random.default_rng(1)
    agent = EidosAgent(seed=1, enable_reasoning=True, apply_hypotheses=True)
    fire, _ = _train_fire(agent, rng)

    agent.workspace.clear()
    agent._recovery_context.clear()

    surprise = rng.normal(0, 8, 64)
    result = agent.step(surprise, "anomaly")

    assert result["recovery_inference_source"] == "none"
    assert result["inferred_recovery_label"] is None

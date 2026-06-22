"""Tests for v3.1 consequential meta-cognition."""

import numpy as np

from agent.eidos import EidosAgent


def _ambiguous_setup(agent: EidosAgent, rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(0, 1, 64)
    alpha = base.copy()
    beta = base + rng.normal(0, 0.02, 64)
    agent.register_concept("alpha", alpha)
    agent.register_concept("beta", beta)
    for _ in range(30):
        agent.step(alpha + rng.normal(0, 0.04, 64), "alpha")
    for _ in range(8):
        agent.step(beta + rng.normal(0, 0.04, 64), "beta")
    return base


def test_v31_defers_ambiguous_hypothesis():
    rng = np.random.default_rng(0)
    agent = EidosAgent(seed=0, enable_meta_consequential=True)
    base = _ambiguous_setup(agent, rng)
    agent.enable_reasoning = True
    result = agent.step(base + rng.normal(0, 4, 64), "anomaly")
    assert "ambiguous_hypothesis" in result["meta_cognition_flags"]
    assert "hypothesis_deferred" in result["meta_cognition_flags"]
    assert not result["hypothesis_applied"]


def test_v30_observational_still_applies():
    rng = np.random.default_rng(1)
    agent = EidosAgent(
        seed=1, enable_meta_consequential=False, enable_meta_cognition=True
    )
    base = _ambiguous_setup(agent, rng)
    agent.enable_reasoning = True
    result = agent.step(base + rng.normal(0, 4, 64), "anomaly")
    assert "ambiguous_hypothesis" in result["meta_cognition_flags"]
    assert result["hypothesis_applied"]

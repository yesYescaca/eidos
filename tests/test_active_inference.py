"""Tests for v4.0 active inference."""

import numpy as np

from agent.eidos import EidosAgent
from architecture.components.active_inference import ActiveInferenceController


def test_active_inference_selects_probe_with_goal():
    rng = np.random.default_rng(0)
    controller = ActiveInferenceController()
    concepts = {
        "fire": rng.normal(0, 1, 64),
        "water": rng.normal(0, 1, 64) + 3,
    }
    agent = EidosAgent(seed=0, enable_active_inference=False)
    for name, vec in concepts.items():
        agent.register_concept(name, vec)
    observation = 0.5 * concepts["fire"] + 0.5 * concepts["water"]
    context = observation[: agent.hidden_dim]
    decision = controller.select_action(
        observation,
        context,
        concepts,
        agent.prediction,
        goal=concepts["fire"],
    )
    selected = decision["action"]
    assert selected["type"] == "probe"
    assert selected["concept"] == "fire"


def test_agent_probes_on_unknown_input():
    rng = np.random.default_rng(1)
    agent = EidosAgent(seed=1, enable_active_inference=True, enable_meta_cognition=False)
    fire = rng.normal(0, 1, 64)
    agent.register_concept("fire", fire)
    for _ in range(15):
        agent.step(fire + rng.normal(0, 0.05, 64), "fire")
    agent.workspace.clear()
    agent._recovery_context.clear()
    agent.surprise._history.clear()
    result = agent.step(rng.normal(0, 8, 64), "anomaly", goal=fire)
    assert result["selected_action"] == "probe:fire"
    assert result["inferred_recovery_label"] == "fire"

"""Tests for metacognitive injection (v7.0)."""

from architecture.gate.gate_policy import GateEvaluation
from architecture.hybrid.metacognitive_prompt import build_meta_injection, build_revision_prompt
from architecture.hybrid.hybrid_agent import HybridEidosAgent
from architecture.hybrid.llm_backend import RoundRobinMockLLM


def test_build_meta_injection_contains_decision():
    evaluation = GateEvaluation(
        decision="clarify",
        cognitive_decision="observe",
        scores={"draft_goal_alignment": 0.5},
        reasons=["draft_goal_misalignment:0.500"],
    )
    block = build_meta_injection(evaluation, {}, {})
    assert "clarify" in block
    assert "0.500" in block or "alignment" in block


def test_meta_injection_revises_and_commits():
    concepts = {
        "alpha": "the reactor core is overheating in sector alpha",
        "beta": "the reactor core is overheating in sector beta",
    }
    wrong = "The reactor core is overheating in sector beta. Recommend immediate cooling protocol."
    right = "The reactor core is overheating in sector alpha. Recommend immediate cooling protocol."

    hybrid = HybridEidosAgent(
        llm=RoundRobinMockLLM([wrong, right]),
        enable_gate=True,
        enable_meta_injection=True,
        hybrid_embedding=False,
        seed=0,
        enable_meta_cognition=False,
        enable_active_inference=False,
    )
    hybrid.register_domain(concepts)
    hybrid.warm_session([("alpha", concepts["alpha"])], n_each=10)

    result = hybrid.respond(
        "the reactor core is overheating in sector unknown",
        goal_text=concepts["alpha"],
    )
    assert len(result["revision_rounds"]) >= 1
    assert "sector alpha" in result["final_response"].lower() or result["gate_decision"] != "commit"

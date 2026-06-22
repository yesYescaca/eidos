"""Tests for hybrid LLM + EIDOS spike."""

from architecture.hybrid.hybrid_agent import HybridEidosAgent, merge_decisions
from architecture.hybrid.llm_backend import MockLanguageModel


def test_merge_decisions_prefers_cautious():
    assert merge_decisions("observe", "commit", "defer") == "defer"
    assert merge_decisions("probe", "clarify") == "clarify"


def test_hybrid_gates_wrong_llm_draft():
    concepts = {
        "alpha": "the reactor core is overheating in sector alpha",
        "beta": "the reactor core is overheating in sector beta",
    }
    hybrid = HybridEidosAgent(
        llm=MockLanguageModel(bias="beta"),
        enable_gate=True,
        seed=0,
        hybrid_embedding=False,
        enable_meta_cognition=True,
        enable_meta_consequential=True,
        enable_active_inference=True,
    )
    hybrid.register_domain(concepts)
    hybrid.warm_session([("alpha", concepts["alpha"])], n_each=30)

    result = hybrid.respond(
        "the reactor core is overheating in sector unknown",
        goal_text=concepts["alpha"],
    )
    assert result["gated"]
    assert result["gate_decision"] in ("defer", "clarify", "probe", "sleep")
    assert "sector beta" in result["llm_draft"]


def test_hybrid_no_gate_commits_draft():
    hybrid = HybridEidosAgent(
        llm=MockLanguageModel(),
        enable_gate=False,
        seed=1,
        hybrid_embedding=False,
    )
    hybrid.register_domain({"fire": "flames in the kitchen"})
    result = hybrid.respond("what is happening", goal_text="flames in the kitchen")
    assert not result["gated"]
    assert result["final_response"] == result["llm_draft"]

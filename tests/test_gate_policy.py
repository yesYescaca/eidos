"""Tests for v6.0 unified GatePolicy."""

from architecture.gate.gate_policy import GatePolicy, gate_response
from architecture.hybrid.hybrid_agent import HybridEidosAgent, merge_decisions
from architecture.hybrid.llm_backend import MockLanguageModel


def test_merge_decisions_prefers_cautious():
    assert merge_decisions("observe", "commit", "defer") == "defer"
    assert merge_decisions("probe", "clarify") == "clarify"


def test_draft_goal_misalignment_triggers_clarify():
    policy = GatePolicy(min_draft_goal_align=0.82)
    question_step = {"surprise_ratio": 0.5, "hypothesis_applied": False}
    draft_step = {"prediction_error": 0.1, "hypothesis_applied": False}

    from architecture.bridge.text_grounding import TextGroundingBridge

    grounding = TextGroundingBridge()
    goal = "the reactor core is overheating in sector alpha"
    draft = "The reactor core is overheating in sector beta. Recommend immediate cooling protocol."

    evaluation = policy.evaluate(
        question_step,
        draft_step,
        user_text="status report please",
        draft_text=draft,
        goal_text=goal,
        grounding=grounding,
        text_concepts={},
    )

    assert evaluation.decision == "clarify"
    assert any(r.startswith("draft_goal_misalignment") for r in evaluation.reasons)


def test_unified_gate_blocks_legacy_commit_path():
    concepts = {
        "alpha": "the reactor core is overheating in sector alpha",
        "beta": "the reactor core is overheating in sector beta",
    }
    hybrid = HybridEidosAgent(
        llm=MockLanguageModel(bias="beta"),
        enable_gate=True,
        use_unified_gate=True,
        seed=0,
        enable_meta_cognition=False,
        enable_meta_consequential=False,
        enable_active_inference=False,
    )
    hybrid.register_domain(concepts)
    hybrid.warm_session([("alpha", concepts["alpha"])], n_each=8)

    result = hybrid.respond(
        "the reactor core is overheating in sector unknown",
        goal_text=concepts["alpha"],
    )
    assert result["gated"]
    assert result["gate_decision"] == "clarify"
    assert result["final_response"] != result["llm_draft"]


def test_gate_response_clarify_message():
    from architecture.gate.gate_policy import GateEvaluation

    msg = gate_response(GateEvaluation("clarify", "clarify"), "draft text")
    assert "clarify" in msg.lower()

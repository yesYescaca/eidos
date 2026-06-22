"""Tests for v7.1 gate draft-concept mismatch."""

from architecture.bridge.text_grounding import TextGroundingBridge
from architecture.gate.gate_policy import GatePolicy


def test_draft_concept_mismatch_triggers_clarify():
    policy = GatePolicy(min_draft_goal_align=0.99)
    grounding = TextGroundingBridge()
    concepts = {
        "alpha": "the reactor core is overheating in sector alpha",
        "beta": "the reactor core is overheating in sector beta",
    }
    goal = concepts["alpha"]
    draft = "The reactor core is overheating in sector beta. Recommend immediate cooling."

    evaluation = policy.evaluate(
        {"surprise_ratio": 0.5, "hypothesis_applied": False},
        {"prediction_error": 0.1, "hypothesis_applied": False},
        user_text="reactor overheating sector unknown",
        draft_text=draft,
        goal_text=goal,
        grounding=grounding,
        text_concepts=concepts,
    )

    assert evaluation.decision == "clarify"
    assert any("draft_concept_mismatch" in r for r in evaluation.reasons)


def test_live_runner_skips_without_api_key(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from eval.eidos_eval.live_runner import main

    code = main([])
    assert code == 0
    assert "SKIP" in capsys.readouterr().out

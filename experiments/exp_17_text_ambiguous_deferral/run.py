"""Experiment 17: Goal-directed text probing (v5.0).

Cold ambiguous phrase + goal text. Active inference selects probe on the
goal-aligned concept; passive observes without probing.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import TEXT_ANOMALY_LABEL
from agent.text_agent import EidosTextAgent


PHRASES = {
    "fire": "smoke and flames spreading through the building",
    "smoke": "thick smoke reducing visibility in the hallway",
    "water": "cold water flooding the basement floor",
}
AMBIGUOUS = "heat and smoke detected near the east wing"


def train(agent: EidosTextAgent) -> None:
    for label, phrase in PHRASES.items():
        agent.register_text_concept(label, phrase)
    for _ in range(50):
        agent.step_text(PHRASES["fire"], "fire")
    for _ in range(15):
        agent.step_text(PHRASES["smoke"], "smoke")


def probe_step(agent: EidosTextAgent, use_goal: bool) -> dict:
    agent.agent.workspace.clear()
    agent.agent._recovery_context.clear()
    agent.agent.surprise._history.clear()
    agent.agent.enable_reasoning = True
    kwargs = {"goal_text": PHRASES["fire"]} if use_goal else {}
    return agent.step_text(AMBIGUOUS, TEXT_ANOMALY_LABEL, **kwargs)


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    passive = EidosTextAgent(
        seed=42,
        enable_active_inference=False,
        enable_meta_cognition=False,
    )
    active = EidosTextAgent(
        seed=42,
        enable_active_inference=True,
        enable_meta_cognition=False,
    )

    train(passive)
    train(active)

    passive_result = probe_step(passive, use_goal=False)
    active_result = probe_step(active, use_goal=True)

    active_probed = active_result.get("text_decision") == "probe"
    active_inferred = active_result.get("inferred_recovery_label") == "fire"
    passive_no_probe = passive_result.get("text_decision") != "probe"

    scenario_pass = bool(active_probed and active_inferred and passive_no_probe)

    results = {
        "experiment": "exp_17_text_goal_probe",
        "description": "v5.0 text + active inference probes goal-aligned concept",
        "passive": {
            "text_decision": passive_result.get("text_decision"),
            "inferred": passive_result.get("inferred_recovery_label"),
            "selected_action": passive_result.get("selected_action"),
        },
        "active": {
            "text_decision": active_result.get("text_decision"),
            "inferred": active_result.get("inferred_recovery_label"),
            "selected_action": active_result.get("selected_action"),
        },
        "checks": {
            "active_probed": active_probed,
            "active_inferred_fire": active_inferred,
            "passive_no_probe": passive_no_probe,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Passive\n(no goal)", "Active\n(goal+probe)"],
        [
            0 if passive_result.get("inferred_recovery_label") == "fire" else 1,
            0 if active_result.get("inferred_recovery_label") == "fire" else 1,
        ],
        color=["#95a5a6", "#3498db"],
    )
    ax.set_ylabel("Inference failure (0 = correct)")
    ax.set_title("Exp 17: Goal-Directed Text Probe (v5.0)")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 17: Goal-Directed Text Probe (v5.0)")
    print("=" * 50)
    print(f"  Passive: decision={passive_result.get('text_decision')}, inferred={passive_result.get('inferred_recovery_label')}")
    print(f"  Active:  decision={active_result.get('text_decision')}, action={active_result.get('selected_action')}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

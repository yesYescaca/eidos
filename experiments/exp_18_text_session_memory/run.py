"""Experiment 18: Text session memory via sleep (v5.0).

Train on fire phrases without sleep; misleading water warmup biases recent
context. Pre-surprise sleep consolidates episodic fire into BeliefGraph.
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
    "water": "cold water flooding the basement floor",
    "smoke": "thick smoke reducing visibility in the hallway",
}
TRAINED = "fire"
SURPRISE = "unexpected heat signature detected near sector seven"


def train_fire_no_sleep(agent: EidosTextAgent, n_steps: int = 70) -> None:
    for label, phrase in PHRASES.items():
        agent.register_text_concept(label, phrase)
    for _ in range(n_steps):
        agent.step_text(PHRASES[TRAINED], TRAINED)


def scenario(agent: EidosTextAgent, pre_surprise_sleep: bool) -> dict:
    agent.agent.workspace.clear()
    agent.agent._recovery_context.clear()
    agent.agent.surprise._history.clear()

    agent.agent.enable_reasoning = False
    for _ in range(14):
        agent.step_text(PHRASES["water"], "water")

    if pre_surprise_sleep:
        agent.sleep()

    agent.agent.enable_reasoning = True
    result = agent.step_text(SURPRISE, TEXT_ANOMALY_LABEL)
    hyp = result.get("hypothesis") or {}
    return {
        "inferred": result.get("inferred_recovery_label"),
        "source": result.get("recovery_inference_source"),
        "associate": hyp.get("associate"),
        "text_decision": result.get("text_decision"),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    misled = EidosTextAgent(
        seed=42,
        enable_meta_cognition=False,
        enable_active_inference=False,
    )
    recovered = EidosTextAgent(
        seed=42,
        enable_meta_cognition=False,
        enable_active_inference=False,
    )

    train_fire_no_sleep(misled)
    train_fire_no_sleep(recovered)

    misled_result = scenario(misled, pre_surprise_sleep=False)
    recovered_result = scenario(recovered, pre_surprise_sleep=True)

    misled_wrong = misled_result["inferred"] != TRAINED
    recovered_right = recovered_result["inferred"] == TRAINED
    scenario_pass = bool(misled_wrong and recovered_right)

    results = {
        "experiment": "exp_18_text_session_memory",
        "description": "v5.0 text + sleep before surprise fixes misleading phrase context",
        "without_sleep": misled_result,
        "with_sleep": recovered_result,
        "checks": {
            "misled_without_sleep": misled_wrong,
            "recovered_with_sleep": recovered_right,
        },
        "pass": scenario_pass,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["No pre-sleep\n(misled)", "Pre-sleep\n(recovered)"],
        [
            0 if misled_result["inferred"] == TRAINED else 1,
            0 if recovered_result["inferred"] == TRAINED else 1,
        ],
        color=["#e74c3c", "#2ecc71"],
    )
    ax.set_ylabel("Inference failure (0 = correct)")
    ax.set_title("Exp 18: Text Session Memory (v5.0)")
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("=" * 50)
    print("EXPERIMENT 18: Text Session Memory (v5.0)")
    print("=" * 50)
    print(f"  No sleep: inferred={misled_result['inferred']}, source={misled_result['source']}")
    print(f"  Sleep:    inferred={recovered_result['inferred']}, source={recovered_result['source']}")
    print(f"PASS: {scenario_pass}")


if __name__ == "__main__":
    main()

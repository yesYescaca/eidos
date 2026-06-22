"""Interactive-style hybrid demo — GPT-2 / Groq + EIDOS gate (optional deps)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from architecture.hybrid.hybrid_agent import HybridEidosAgent
from architecture.hybrid.llm_backend import GPT2LanguageModel, MockLanguageModel
from architecture.hybrid.llm_factory import create_live_llm, live_llm_available


DOMAIN = {
    "fire": "smoke and flames spreading through the building",
    "water": "cold water flooding the basement floor",
    "smoke": "thick smoke reducing visibility in the hallway",
}


def build_llm(use_gpt2: bool, use_groq: bool):
    if use_groq:
        if not live_llm_available("groq"):
            print("GROQ_API_KEY not set — falling back to MockLLM.")
            return MockLanguageModel(bias="beta")
        print("Using Groq live LLM (llama-3.3-70b-versatile default).")
        return create_live_llm("groq")
    if use_gpt2:
        try:
            print("Loading GPT-2 (CPU)... first run downloads ~500MB.")
            return GPT2LanguageModel()
        except ImportError:
            print("transformers not installed — using MockLLM.")
            print("  pip install -r requirements-hybrid.txt")
    return MockLanguageModel(bias="beta")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid QA demo: LLM + EIDOS gate")
    parser.add_argument(
        "--gpt2", action="store_true", help="Use real GPT-2 (requires transformers)"
    )
    parser.add_argument(
        "--groq", action="store_true", help="Use Groq live API (requires GROQ_API_KEY)"
    )
    parser.add_argument(
        "--no-gate", action="store_true", help="Disable EIDOS gate (LLM only)"
    )
    parser.add_argument(
        "--meta-injection",
        action="store_true",
        help="Enable metacognitive revision loop (v7.0)",
    )
    parser.add_argument(
        "--belief-context",
        action="store_true",
        help="Inject EIDOS belief state into LLM prompt (v7.1)",
    )
    args = parser.parse_args()

    hybrid = HybridEidosAgent(
        llm=build_llm(args.gpt2, args.groq),
        enable_gate=not args.no_gate,
        enable_meta_injection=args.meta_injection,
        enable_belief_context=args.belief_context,
        hybrid_embedding=not args.groq,
        seed=42,
        enable_meta_cognition=True,
        enable_meta_consequential=True,
        enable_active_inference=True,
    )
    hybrid.register_domain(DOMAIN)
    hybrid.warm_session([("fire", DOMAIN["fire"])], n_each=25)

    scenarios = [
        ("Clear", "fire alarm triggered in hallway", DOMAIN["fire"]),
        ("Ambiguous", "heat and smoke detected near the east wing", DOMAIN["fire"]),
    ]

    print("=" * 60)
    print("HYBRID DEMO — LLM proposes, EIDOS gates")
    print("=" * 60)

    for name, question, goal in scenarios:
        result = hybrid.respond(question, goal_text=goal)
        print(f"\n--- {name} ---")
        print(f"Q: {question}")
        print(f"Gate: {result['gate_decision']} (gated={result['gated']})")
        if result.get("revision_rounds"):
            print(f"Revisions: {len(result['revision_rounds'])}")
        print(f"Draft: {result['llm_draft'][:120]}...")
        print(f"Final: {result['final_response'][:200]}")


if __name__ == "__main__":
    main()

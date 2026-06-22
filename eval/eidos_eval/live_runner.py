"""Live EIDOS-Eval — Groq API comparison (v7.1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from architecture.hybrid.hybrid_agent import HybridEidosAgent
from architecture.hybrid.llm_backend import LanguageModelBackend
from architecture.hybrid.llm_factory import create_live_llm, live_llm_available
from eval.eidos_eval.runner import DEFAULT_QUESTIONS_PATH, EidosEvalHarness, EvalMode

LIVE_QUESTIONS_PATH = Path(__file__).resolve().parent / "questions_live.json"


def run_live_comparison(
    *,
    provider: str = "groq",
    questions_path: Path | None = None,
    seed: int = 42,
    modes: list[EvalMode] | None = None,
    llm: LanguageModelBackend | None = None,
) -> dict:
    """Run EIDOS-Eval with a live LLM for all modes."""
    path = questions_path or LIVE_QUESTIONS_PATH
    harness = EidosEvalHarness(path)
    backend = llm or create_live_llm(provider)  # type: ignore[arg-type]

    def hybrid_factory(**kwargs: object) -> HybridEidosAgent:
        kwargs["llm"] = backend
        return HybridEidosAgent(**kwargs)

    selected = modes or list(EvalMode)
    return {
        mode.value: harness.run_mode(
            mode,
            seed=seed,
            hybrid_factory=hybrid_factory,
            live_llm=backend,
        )
        for mode in selected
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EIDOS-Eval live API run (Groq default)")
    parser.add_argument(
        "--provider",
        choices=("groq", "openai"),
        default="groq",
        help="Live LLM provider (default: groq)",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=LIVE_QUESTIONS_PATH,
        help="Question set JSON path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[m.value for m in EvalMode],
        default=[m.value for m in EvalMode],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report to path",
    )
    args = parser.parse_args(argv)

    if not live_llm_available(args.provider):
        key_name = "GROQ_API_KEY" if args.provider == "groq" else "OPENAI_API_KEY"
        print(f"SKIP: {key_name} not set — live eval unavailable.")
        return 0

    modes = [EvalMode(m) for m in args.modes]
    reports = run_live_comparison(
        provider=args.provider,
        questions_path=args.questions,
        seed=args.seed,
        modes=modes,
    )

    print("=" * 55)
    print(f"EIDOS-EVAL LIVE ({args.provider}) — v7.1")
    print("=" * 55)
    payload: dict = {}
    for mode, report in reports.items():
        print(
            f"  [{mode}] acc={report.accuracy:.1%} "
            f"commit_acc={report.accuracy_when_commit:.1%} "
            f"abstain={report.abstention_rate:.1%} "
            f"false_commit={report.false_commit_rate:.1%} "
            f"must_abstain_safe={report.must_abstain_safe_rate:.1%}"
        )
        payload[mode] = report.to_dict()

    if args.out:
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nReport saved to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

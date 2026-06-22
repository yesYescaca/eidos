"""Live EIDOS-Eval — Groq API comparison (v7.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from architecture.bridge.embedding_factory import create_live_grounding, resolve_active_backend
from architecture.hybrid.hybrid_agent import HybridEidosAgent
from architecture.hybrid.llm_backend import LanguageModelBackend
from architecture.hybrid.llm_factory import create_live_llm, live_llm_available
from eval.eidos_eval.llm_cache import CachedLLM, DEFAULT_CACHE_PATH
from eval.eidos_eval.runner import EidosEvalHarness, EvalMode

LIVE_QUESTIONS_PATH = Path(__file__).resolve().parent / "questions_live.json"

DEFAULT_LIVE_MODES = [
    EvalMode.LLM_ALONE,
    EvalMode.EIDOS_GATE,
    EvalMode.EIDOS_BELIEF,
    EvalMode.EIDOS_META,
]


def run_live_comparison(
    *,
    provider: str = "groq",
    questions_path: Path | None = None,
    seed: int = 42,
    modes: list[EvalMode] | None = None,
    llm: LanguageModelBackend | None = None,
    use_cache: bool = True,
    cache_path: Path | None = None,
    embedding: str = "auto",
) -> dict:
    """Run EIDOS-Eval with a live LLM for all modes."""
    path = questions_path or LIVE_QUESTIONS_PATH
    harness = EidosEvalHarness(path)
    backend = llm or create_live_llm(provider)  # type: ignore[arg-type]
    if use_cache:
        backend = CachedLLM(backend, cache_path or DEFAULT_CACHE_PATH)

    prefer_sbert = embedding != "hash"
    if embedding == "sbert":
        prefer_sbert = True
    shared_grounding = create_live_grounding(prefer_sbert=prefer_sbert)
    backend_label = resolve_active_backend(shared_grounding)

    def hybrid_factory(**kwargs: object) -> HybridEidosAgent:
        kwargs["llm"] = backend
        kwargs["grounding"] = shared_grounding
        kwargs["hybrid_embedding"] = False
        return HybridEidosAgent(**kwargs)

    selected = modes or DEFAULT_LIVE_MODES
    reports = harness.run_comparison(
        seed=seed,
        hybrid_factory=hybrid_factory,
        live_llm=backend,
        modes=selected,
    )
    reports["_embedding_backend"] = backend_label  # type: ignore[assignment]
    return reports


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
        default=[m.value for m in DEFAULT_LIVE_MODES],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report to path",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable live response disk cache",
    )
    parser.add_argument(
        "--embedding",
        choices=("auto", "hash", "sbert"),
        default="auto",
        help="Embedding backend for gate (auto=SBERT with hash fallback)",
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
        use_cache=not args.no_cache,
        embedding=args.embedding,
    )
    emb = reports.pop("_embedding_backend", "unknown")
    summary = EidosEvalHarness.summarize_comparison(reports)

    print("=" * 55)
    print(f"EIDOS-EVAL LIVE ({args.provider}) — v7.2")
    print(f"Embedding: {emb}")
    print("=" * 55)
    payload: dict = {"embedding_backend": emb, "reports": {}, "summary": summary.to_dict()}
    for mode, report in reports.items():
        print(
            f"  [{mode}] task_acc={report.task_accuracy:.1%} "
            f"commit_acc={report.accuracy_when_commit:.1%} "
            f"abstain={report.abstention_rate:.1%} "
            f"false_commit={report.false_commit_rate:.1%} "
            f"must_abstain_safe={report.must_abstain_safe_rate:.1%}"
        )
        payload["reports"][mode] = report.to_dict()

    print("  ---")
    print(
        f"  Δ commit_acc (gate vs alone): {summary.selective_accuracy_delta_gate:+.1%}"
    )
    print(
        f"  Δ task_acc (gate vs alone): {summary.task_accuracy_delta_gate:+.1%}"
    )
    print(
        f"  false_commit reduction (gate): {summary.false_commit_reduction_gate:.1%}"
    )

    if args.out:
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nReport saved to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

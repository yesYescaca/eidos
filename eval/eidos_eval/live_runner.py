"""Live EIDOS-Eval — Groq API comparison (v7.5)."""

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
from eval.eidos_eval.runner import EidosEvalHarness, EvalMode, LIVE_COMPARISON_MODES

LIVE_QUESTIONS_PATH = Path(__file__).resolve().parent / "questions_live.json"
TRUTHFULQA_50_PATH = Path(__file__).resolve().parent / "questions_truthfulqa_50.json"
MIXED_50_PATH = Path(__file__).resolve().parent / "questions_mixed_50.json"

DEFAULT_LIVE_MODES = list(LIVE_COMPARISON_MODES)


def _print_report(mode: str, report: object) -> None:
    """Print mode summary — TruthfulQA / mixed metrics when available."""
    r = report
    line = (
        f"  [{mode}] task_acc={r.task_accuracy:.1%} "
        f"commit_acc={r.accuracy_when_commit:.1%} "
        f"abstain={r.abstention_rate:.1%}"
    )
    if getattr(r, "grading_mode", None) in ("truthfulqa", "mixed"):
        line += (
            f" miscon_TI={r.misconception_ti_rate:.1%}"
            f" commit_TI={r.misconception_commit_ti_rate:.1%}"
        )
        if r.grading_mode == "truthfulqa":
            line += (
                f" TI={r.truthful_informative_rate:.1%}"
                f" misc={r.misconception_commit_rate:.1%}"
            )
        if r.grading_mode == "mixed" and r.ambiguous_safe_rate is not None:
            line += f" ambig_safe={r.ambiguous_safe_rate:.1%}"
    else:
        line += (
            f" false_commit={r.false_commit_rate:.1%} "
            f"must_abstain_safe={r.must_abstain_safe_rate:.1%}"
        )
    print(line)


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
    limit: int | None = None,
) -> dict:
    """Run EIDOS-Eval with a live LLM for all modes."""
    path = questions_path or LIVE_QUESTIONS_PATH
    harness = EidosEvalHarness(path, limit=limit)
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
    reports["_grading_mode"] = harness.grading_mode  # type: ignore[assignment]
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only first N questions (debug)",
    )
    parser.add_argument(
        "--truthfulqa",
        action="store_true",
        help="Use TruthfulQA Misconceptions N=50 set",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Use mixed N=50 (25 misconceptions + 25 ambiguous)",
    )
    args = parser.parse_args(argv)

    if not live_llm_available(args.provider):
        key_name = "GROQ_API_KEY" if args.provider == "groq" else "OPENAI_API_KEY"
        print(f"SKIP: {key_name} not set — live eval unavailable.")
        return 0

    modes = [EvalMode(m) for m in args.modes]
    if args.mixed:
        questions_path = MIXED_50_PATH
    elif args.truthfulqa:
        questions_path = TRUTHFULQA_50_PATH
    else:
        questions_path = args.questions

    default_out = None
    if args.out is None:
        if args.mixed:
            default_out = Path(__file__).resolve().parent / "live_mixed_report.json"
        elif args.truthfulqa:
            default_out = Path(__file__).resolve().parent / "live_truthfulqa_report.json"

    reports = run_live_comparison(
        provider=args.provider,
        questions_path=questions_path,
        seed=args.seed,
        modes=modes,
        use_cache=not args.no_cache,
        embedding=args.embedding,
        limit=args.limit,
    )
    emb = reports.pop("_embedding_backend", "unknown")
    grading_mode = reports.pop("_grading_mode", None)
    summary = EidosEvalHarness.summarize_comparison(reports)

    print("=" * 55)
    print(f"EIDOS-EVAL LIVE ({args.provider}) — v7.5")
    print(f"Embedding: {emb}")
    if grading_mode:
        print(f"Grading: {grading_mode}")
    print("=" * 55)
    payload: dict = {
        "embedding_backend": emb,
        "grading_mode": grading_mode,
        "reports": {},
        "summary": summary.to_dict(),
    }
    for mode, report in reports.items():
        _print_report(mode, report)
        payload["reports"][mode] = report.to_dict()

    print("  ---")
    print(
        f"  Δ commit_acc (gate vs alone): {summary.selective_accuracy_delta_gate:+.1%}"
    )
    print(
        f"  Δ task_acc (belief vs alone): {summary.task_accuracy_delta_belief:+.1%}"
    )
    print(
        f"  Δ task_acc (CoT vs alone): {summary.task_accuracy_delta_cot:+.1%}"
    )
    print(f"  Belief beats CoT (task): {summary.belief_beats_cot}")
    if summary.truthful_informative_delta_belief is not None:
        print(
            f"  Δ TI (belief vs alone): {summary.truthful_informative_delta_belief:+.1%}"
        )
        print(
            f"  Δ TI (CoT vs alone): {summary.truthful_informative_delta_cot:+.1%}"
        )
        print(f"  Belief beats CoT (TI): {summary.belief_beats_cot_ti}")
    if summary.misconception_commit_ti_belief is not None:
        print(
            f"  Miscon commit TI (belief): {summary.misconception_commit_ti_belief:.1%}"
        )
        print(
            f"  Miscon commit TI (CoT): {summary.misconception_commit_ti_cot:.1%}"
        )
        print(
            f"  Belief beats CoT (miscon commits): "
            f"{summary.belief_beats_cot_misconception_commits}"
        )
    print(
        f"  false_commit reduction (gate): {summary.false_commit_reduction_gate:.1%}"
    )

    out_path = args.out or default_out
    if out_path:
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nReport saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

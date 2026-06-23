"""Run EIDOS-Eval live across multiple LLMs (v7.6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from architecture.hybrid.llm_factory import live_llm_available
from eval.eidos_eval.live_models import (
    GROQ_EVAL_MODELS,
    default_models_for_provider,
    model_slug,
    report_basename,
    resolve_model_id,
)
from eval.eidos_eval.live_runner import (
    MIXED_50_PATH,
    REPORTS_DIR,
    TRUTHFULQA_50_PATH,
    build_live_payload,
    print_live_summary,
    run_live_comparison,
)

BENCHMARK_PATHS = {
    "truthfulqa": TRUTHFULQA_50_PATH,
    "mixed": MIXED_50_PATH,
}


def _belief_beats_cot_commit_ti(payload: dict) -> bool | None:
    belief = payload["reports"].get("eidos_belief", {})
    cot = payload["reports"].get("llm_cot", {})
    b = belief.get("misconception_commit_ti_rate")
    c = cot.get("misconception_commit_ti_rate")
    if b is None or c is None:
        return None
    return b > c


def run_multimodel_eval(
    *,
    provider: str = "groq",
    models: list[str],
    benchmarks: list[str],
    seed: int = 42,
    use_cache: bool = True,
    embedding: str = "auto",
    limit: int | None = None,
) -> dict:
    """Run each benchmark for each model; return combined summary."""
    results: dict = {
        "provider": provider,
        "models": models,
        "benchmarks": benchmarks,
        "runs": [],
    }
    for model in models:
        model_id = resolve_model_id(provider, model)
        for bench in benchmarks:
            questions_path = BENCHMARK_PATHS[bench]
            try:
                reports = run_live_comparison(
                    provider=provider,
                    model=model,
                    questions_path=questions_path,
                    seed=seed,
                    use_cache=use_cache,
                    embedding=embedding,
                    limit=limit,
                )
            except Exception as exc:  # noqa: BLE001 — continue other model/benchmark pairs
                err_row = {
                    "model": model_id,
                    "benchmark": bench,
                    "error": str(exc),
                    "belief_beats_cot_commit_ti": None,
                }
                results["runs"].append(err_row)
                results.setdefault("errors", []).append(err_row)
                print()
                print(f"=== {model_id} / {bench} — FAILED ===")
                print(f"  {exc}")
                continue

            payload = build_live_payload(reports)
            out_path = REPORTS_DIR / report_basename(bench, model_id)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2))

            belief_beats = _belief_beats_cot_commit_ti(payload)
            belief = payload["reports"].get("eidos_belief", {})
            cot = payload["reports"].get("llm_cot", {})
            alone = payload["reports"].get("llm_alone", {})
            row = {
                "model": model_id,
                "benchmark": bench,
                "report_path": str(out_path),
                "belief_task_acc": belief.get("task_accuracy"),
                "belief_commit_ti": belief.get("misconception_commit_ti_rate"),
                "cot_commit_ti": cot.get("misconception_commit_ti_rate"),
                "alone_task_acc": alone.get("task_accuracy"),
                "belief_beats_cot_commit_ti": belief_beats,
            }
            results["runs"].append(row)

            print()
            print(f"=== {model_id} / {bench} ===")
            print_live_summary(payload, provider=provider, model_id=model_id)
            if belief_beats is not None:
                print(f"  Belief beats CoT (commit TI): {belief_beats}")
    return results


def print_comparison_table(results: dict) -> None:
    print()
    print("=" * 72)
    print("MULTI-MODEL COMPARISON (belief commit TI vs CoT commit TI)")
    print("=" * 72)
    print(f"{'Model':<32} {'Bench':<12} {'Belief':>8} {'CoT':>8} {'Alone':>8} {'B>C':>5}")
    print("-" * 72)
    for row in results["runs"]:
        if row.get("error"):
            slug = model_slug(row["model"])
            err = row["error"][:40] + "…" if len(row["error"]) > 40 else row["error"]
            print(f"{slug:<32} {row['benchmark']:<12} {'ERROR':>8} {'—':>8} {'—':>8} {'—':>5}")
            print(f"  ↳ {err}")
            continue
        b = row["belief_commit_ti"]
        c = row["cot_commit_ti"]
        a = row["alone_task_acc"]
        flag = (
            "yes"
            if row["belief_beats_cot_commit_ti"]
            else "no"
            if row["belief_beats_cot_commit_ti"] is False
            else "—"
        )
        b_s = f"{b:.1%}" if b is not None else "—"
        c_s = f"{c:.1%}" if c is not None else "—"
        a_s = f"{a:.1%}" if a is not None else "—"
        slug = model_slug(row["model"])
        print(f"{slug:<32} {row['benchmark']:<12} {b_s:>8} {c_s:>8} {a_s:>8} {flag:>5}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run TruthfulQA + mixed live eval across multiple models (v7.6)"
    )
    parser.add_argument("--provider", choices=("groq", "openai"), default="groq")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model IDs (default: provider eval set)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=tuple(BENCHMARK_PATHS),
        default=["truthfulqa", "mixed"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--embedding", choices=("auto", "hash", "sbert"), default="auto")
    parser.add_argument("--limit", type=int, default=None, help="First N questions only")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPORTS_DIR / "multimodel_summary.json",
    )
    args = parser.parse_args(argv)

    if not live_llm_available(args.provider):
        key_name = "GROQ_API_KEY" if args.provider == "groq" else "OPENAI_API_KEY"
        print(f"SKIP: {key_name} not set — multimodel eval unavailable.")
        return 0

    models = args.models or list(default_models_for_provider(args.provider))
    if args.provider == "groq" and args.models is None:
        models = list(GROQ_EVAL_MODELS)

    results = run_multimodel_eval(
        provider=args.provider,
        models=models,
        benchmarks=args.benchmarks,
        seed=args.seed,
        use_cache=not args.no_cache,
        embedding=args.embedding,
        limit=args.limit,
    )
    print_comparison_table(results)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved to {args.summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

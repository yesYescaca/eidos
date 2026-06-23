"""Compare two live eval reports (e.g. cached vs --no-cache ablation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METRICS = (
    "task_accuracy",
    "truthful_informative_rate",
    "misconception_commit_ti_rate",
    "ambiguous_safe_rate",
    "abstention_rate",
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def compare_reports(
    baseline: dict,
    variant: dict,
    *,
    baseline_label: str = "cached",
    variant_label: str = "no-cache",
    metrics: tuple[str, ...] = ("task_accuracy",),
) -> dict:
    modes = sorted(set(baseline.get("reports", {})) & set(variant.get("reports", {})))
    out: dict = {
        "baseline_label": baseline_label,
        "variant_label": variant_label,
        "baseline_model": baseline.get("model"),
        "variant_model": variant.get("model"),
        "grading_mode": baseline.get("grading_mode") or variant.get("grading_mode"),
        "modes": {},
    }
    for mode in modes:
        br = baseline["reports"][mode]
        vr = variant["reports"][mode]
        mode_out: dict = {}
        for metric in metrics:
            b = br.get(metric)
            v = vr.get(metric)
            if b is None and v is None:
                continue
            delta = (v - b) if b is not None and v is not None else None
            mode_out[metric] = {
                "baseline": b,
                "variant": v,
                "delta": delta,
                "delta_pts": round(delta * 100, 1) if delta is not None else None,
            }
        out["modes"][mode] = mode_out
    return out


def _fmt_rate(x: float | None) -> str:
    return f"{x:.1%}" if x is not None else "—"


def print_comparison(summary: dict, *, metrics: tuple[str, ...]) -> None:
    bl = summary["baseline_label"]
    vl = summary["variant_label"]
    print(f"\nAblation: {bl} vs {vl} ({summary.get('grading_mode', '?')})")
    for metric in metrics:
        short = metric.replace("_rate", "").replace("_accuracy", "")
        print(f"\n  [{short}]")
        print(f"  {'Mode':<18} {bl:>12} {vl:>12} {'Δ pts':>8}")
        print(f"  {'-'*52}")
        for mode, row in summary["modes"].items():
            cell = row.get(metric, {})
            b = _fmt_rate(cell.get("baseline"))
            v = _fmt_rate(cell.get("variant"))
            d = cell.get("delta_pts", "—")
            print(f"  {mode:<18} {b:>12} {v:>12} {str(d):>8}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare cached vs no-cache live reports")
    parser.add_argument("baseline", type=Path, help="Baseline report (e.g. cached run)")
    parser.add_argument("variant", type=Path, help="Variant report (e.g. --no-cache run)")
    parser.add_argument("--baseline-label", default="cached")
    parser.add_argument("--variant-label", default="no-cache")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["task_accuracy"],
        choices=METRICS,
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.baseline.is_file() or not args.variant.is_file():
        print("Both report paths must exist.", file=sys.stderr)
        return 1

    summary = compare_reports(
        _load(args.baseline),
        _load(args.variant),
        baseline_label=args.baseline_label,
        variant_label=args.variant_label,
        metrics=tuple(args.metrics),
    )
    print_comparison(summary, metrics=tuple(args.metrics))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

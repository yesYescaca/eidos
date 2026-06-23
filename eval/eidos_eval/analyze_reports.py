"""CLI: Wilson CIs and paired bootstrap over live EIDOS-Eval report JSONs (v7.8)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.eidos_eval.stats import analyze_report_payload, compare_modes_paired


def _load_report(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _print_summary(path: Path, summary: dict) -> None:
    print(f"\n=== {path.name} ({summary.get('model', '?')}, {summary.get('grading_mode', '?')}) ===")
    for mode, mode_data in summary.get("modes", {}).items():
        parts = [f"{mode}:"]
        for key, val in mode_data.items():
            if key == "n_questions":
                continue
            if isinstance(val, dict) and "formatted" in val:
                parts.append(f"  {key}: {val['formatted']}")
        print("  " + " | ".join(parts[:1] + [p.strip() for p in parts[1:]]))
        for key, val in mode_data.items():
            if key == "n_questions":
                continue
            if isinstance(val, dict) and "formatted" in val:
                print(f"    {key}: {val['formatted']} (k={val['k']}, n={val['n']})")
    for cmp in summary.get("comparisons", []):
        print(
            f"  Paired {cmp['label']} ({cmp['metric']}): "
            f"{cmp['formatted_diff']}  McNemar p={cmp.get('mcmemar_p')}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze live EIDOS-Eval reports (Wilson CIs, paired bootstrap)"
    )
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="Report JSON paths (default: eval/eidos_eval/reports/live_*_report.json)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("MODE_A", "MODE_B"),
        help="Paired bootstrap comparison between two modes",
    )
    parser.add_argument(
        "--metric",
        default="task_correct",
        help="Per-question field for --compare (default: task_correct)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write combined JSON summary to path",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    report_dir = root / "eval" / "eidos_eval" / "reports"
    paths = list(args.reports)
    if not paths:
        paths = sorted(report_dir.glob("live_*_report.json"))

    if not paths:
        print("No report files found.", file=sys.stderr)
        return 1

    combined: dict[str, dict] = {}
    for path in paths:
        if not path.is_file():
            print(f"SKIP missing: {path}", file=sys.stderr)
            continue
        payload = _load_report(path)
        summary = analyze_report_payload(payload)
        combined[str(path)] = summary
        _print_summary(path, summary)

        if args.compare:
            mode_a, mode_b = args.compare
            result = compare_modes_paired(
                payload, mode_a, mode_b, metric=args.metric
            )
            if result:
                print(
                    f"\n  Custom compare {mode_a} vs {mode_b}: "
                    f"{result.mean_diff:+.1%} "
                    f"[{result.ci_low:+.1%}, {result.ci_high:+.1%}]"
                )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

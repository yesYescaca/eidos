#!/usr/bin/env python3
"""Patch Figure 8 in the research paper from no-cache ablation reports."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "docs" / "EIDOS_Research_Paper.html"
CACHED = ROOT / "eval/eidos_eval/reports/live_mixed_llama-3.3-70b-versatile_report.json"
NOCACHE = ROOT / "eval/eidos_eval/reports/live_mixed_llama-3.3-70b-versatile_nocache_report.json"
MODES = ("llm_alone", "llm_cot", "llm_reflection", "eidos_belief")


def _task_rates(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [round(data["reports"][m]["task_accuracy"] * 100) for m in MODES]


def main() -> int:
    if not CACHED.is_file() or not NOCACHE.is_file():
        print("Missing cached or no-cache report JSON.", file=sys.stderr)
        return 1
    cached = _task_rates(CACHED)
    nocache = _task_rates(NOCACHE)
    text = PAPER.read_text(encoding="utf-8")

    cached_pat = r"(label: 'Cached \(disk\)',[\s\S]*?data: )\[[^\]]*\]"
    nocache_pat = r"(label: 'No-cache \(fresh Groq\)',[\s\S]*?data: )\[[^\]]*\]"
    text, n1 = re.subn(cached_pat, rf"\g<1>{cached}", text, count=1)
    text, n2 = re.subn(nocache_pat, rf"\g<1>{nocache}", text, count=1)
    if n1 != 1 or n2 != 1:
        print("Could not find Figure 8 chart data arrays in paper HTML.", file=sys.stderr)
        return 1

    # Table 13 mode rows (task acc % only)
    for mode, c_val, n_val in zip(MODES, cached, nocache, strict=True):
        label = {
            "llm_alone": "LLM alone",
            "llm_cot": "LLM CoT",
            "llm_reflection": "LLM reflection",
            "eidos_belief": "EIDOS belief",
        }[mode]
        row_pat = (
            rf"(<tr><td class=\"td-mode\">{re.escape(label)}</td><td>)\d+%(</td><td>)\d+%(</td><td>)[^<]*(</td></tr>)"
        )
        delta = n_val - c_val
        sign = "+" if delta > 0 else ""
        text, n = re.subn(row_pat, rf"\g<1>{c_val}%\2{n_val}%\3{sign}{delta} pts\4", text, count=1)
        if n != 1:
            print(f"Warning: could not update table row for {label}", file=sys.stderr)

    PAPER.write_text(text, encoding="utf-8")
    print(f"Updated Figure 8 + Table 13: cached={cached}, no-cache={nocache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

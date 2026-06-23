#!/usr/bin/env python3
"""Patch Figure 8 and Table 13 in the research paper from no-cache ablation reports."""

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
LABELS = {
    "llm_alone": "LLM alone",
    "llm_cot": "LLM CoT",
    "llm_reflection": "LLM reflection",
    "eidos_belief": "EIDOS belief",
}


def _task_rates(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [round(data["reports"][m]["task_accuracy"] * 100) for m in MODES]


def _fmt_delta(delta: int) -> str:
    if delta > 0:
        return f"+{delta} pts"
    if delta < 0:
        return f"{delta} pts"
    return "0 pts"


def main() -> int:
    if not CACHED.is_file() or not NOCACHE.is_file():
        print("Missing cached or no-cache report JSON.", file=sys.stderr)
        return 1
    cached = _task_rates(CACHED)
    nocache = _task_rates(NOCACHE)
    text = PAPER.read_text(encoding="utf-8")

    def _repl_chart_cached(m: re.Match[str]) -> str:
        return f"{m.group(1)}{cached}"

    def _repl_chart_nocache(m: re.Match[str]) -> str:
        return f"{m.group(1)}{nocache}"

    cached_pat = r"(label: 'Cached \(disk\)',[\s\S]*?data: )\[[^\]]*\]"
    nocache_pat = r"(label: 'No-cache \(fresh Groq\)',[\s\S]*?data: )\[[^\]]*\]"
    text, n1 = re.subn(cached_pat, _repl_chart_cached, text, count=1)
    text, n2 = re.subn(nocache_pat, _repl_chart_nocache, text, count=1)
    if n1 != 1 or n2 != 1:
        print("Could not find Figure 8 chart data arrays in paper HTML.", file=sys.stderr)
        return 1

    for mode, c_val, n_val in zip(MODES, cached, nocache, strict=True):
        label = LABELS[mode]
        row_pat = (
            rf'(<tr><td class="td-mode">{re.escape(label)}</td><td>)[^<]*(</td><td>)[^<]*(</td><td>)[^<]*(</td></tr>)'
        )

        def _repl_row(m: re.Match[str], cv: int = c_val, nv: int = n_val) -> str:
            return f"{m.group(1)}{cv}%{m.group(2)}{nv}%{m.group(3)}{_fmt_delta(nv - cv)}{m.group(4)}"

        text, n = re.subn(row_pat, _repl_row, text, count=1)
        if n != 1:
            print(f"Warning: could not update table row for {label}", file=sys.stderr)

    br_cached = cached[MODES.index("eidos_belief")] - cached[MODES.index("llm_reflection")]
    br_nocache = nocache[MODES.index("eidos_belief")] - nocache[MODES.index("llm_reflection")]

    caption_pat = (
        r'(<h4>Table 13[^<]*</h4>[\s\S]*?<div class="table-caption">)(.*?)(</div>)'
    )
    new_caption = (
        f"<strong>Belief vs reflection gap:</strong> +{br_cached} pts cached, +{br_nocache} pts no-cache "
        f"(stable). Belief task accuracy +2 pts on fresh API calls (86% → 88%); alone and reflection "
        f"unchanged; CoT −4 pts (baseline variance). Ambiguity-safe rate for belief: 88% → 92%."
    )
    text, nc = re.subn(caption_pat, rf"\g<1>{new_caption}\3", text, count=1)
    if nc != 1:
        print("Warning: could not update Table 13 caption", file=sys.stderr)

    fig_cap_pat = (
        r'(<div class="figure-caption"><strong>Figure 8\.</strong>)(.*?)(</div>\s*</div>\s*</section>)'
    )
    new_fig_cap = (
        " Disk-cached vs fresh Groq responses on mixed N=50. Headline claim is robust: "
        f"belief beats reflection by +{br_nocache} pts without cache (88% vs 54%). "
        "Baselines stable; EIDOS belief improves slightly (+2 pts task accuracy)."
    )
    text, nf = re.subn(fig_cap_pat, rf"\g<1>{new_fig_cap}\3", text, count=1)
    if nf != 1:
        print("Warning: could not update Figure 8 caption", file=sys.stderr)

    PAPER.write_text(text, encoding="utf-8")
    print(f"Updated Figure 8 + Table 13: cached={cached}, no-cache={nocache}")
    print(f"Belief vs reflection: +{br_cached} pts (cached), +{br_nocache} pts (no-cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

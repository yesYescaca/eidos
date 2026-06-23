#!/usr/bin/env python3
"""Analyze EIDOS live reports from the repo root (fixes ModuleNotFoundError: eval).

Usage (from any directory):
    py "C:\\path\\to\\eidos\\run_analyze_reports.py" --out eval/eidos_eval/reports/stats_summary.json

Or use the PowerShell wrapper:
    & "C:\\path\\to\\eidos\\run_analyze_reports.ps1"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eidos_eval.analyze_reports import main

if __name__ == "__main__":
    raise SystemExit(main())

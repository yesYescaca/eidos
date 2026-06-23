#!/usr/bin/env python3
"""Run EIDOS live eval from the repo root (fixes ModuleNotFoundError: eval).

Usage (from any directory):
    py "C:\\path\\to\\eidos\\run_live_eval.py" --provider groq --mixed

Or cd into the eidos folder first:
    cd "C:\\path\\to\\eidos"
    py run_live_eval.py --provider groq --mixed
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eidos_eval.live_runner import main

if __name__ == "__main__":
    raise SystemExit(main())

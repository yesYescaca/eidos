#!/usr/bin/env python3
"""Compare two live eval reports from repo root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eidos_eval.compare_ablation import main

if __name__ == "__main__":
    raise SystemExit(main())

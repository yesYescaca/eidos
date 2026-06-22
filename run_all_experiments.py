#!/usr/bin/env python3
"""Run all EIDOS experiments and print a summary table."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [
    ("01", "exp_01_basic_prediction", "Basic prediction"),
    ("02", "exp_02_memory_consolidation", "Memory consolidation"),
    ("03", "exp_03_reasoning_chain", "Reasoning chain"),
    ("04", "exp_04_reasoning_ablation", "Reasoning ablation"),
    ("05", "exp_05_relational_inference", "Relational inference"),
    ("06", "exp_06_reasoning_recovery", "Reasoning recovery"),
    ("07", "exp_07_multi_concept", "Multi-concept"),
    ("08", "exp_08_autonomous_recovery", "Autonomous recovery"),
    ("09", "exp_09_misleading_context", "Misleading context (failure)"),
    ("10", "exp_10_cold_start", "Cold start (failure)"),
    ("11", "exp_11_cls_recovery", "CLS recovery (v2.0)"),
    ("12", "exp_12_meta_misleading_context", "Meta misleading context (v3.0 A)"),
    ("13", "exp_13_meta_ambiguous_reasoning", "Meta ambiguous reasoning (v3.0 B)"),
]


def parse_pass(stdout: str) -> bool | None:
    matches = re.findall(r"PASS[^:]*:\s*(True|False)", stdout, re.IGNORECASE)
    if matches:
        return matches[-1].lower() == "true"
    return None


def read_json_pass(path: Path) -> bool | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "pass" in data:
        return bool(data["pass"])
    if isinstance(data, list):
        return all(item.get("pass", False) for item in data)
    return None


def run_experiment(num: str, folder: str, name: str) -> dict:
    script = ROOT / "experiments" / folder / "run.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    stdout = result.stdout + result.stderr
    passed = parse_pass(stdout)
    if passed is None:
        json_path = ROOT / "experiments" / folder / "results.json"
        passed = read_json_pass(json_path)
    return {
        "num": num,
        "name": name,
        "folder": folder,
        "passed": passed,
        "exit_code": result.returncode,
        "ok": result.returncode == 0 and passed is True,
    }


def main() -> int:
    print("=" * 60)
    print("EIDOS — Full Experiment Suite")
    print("=" * 60)

    rows = []
    for num, folder, name in EXPERIMENTS:
        print(f"\n>>> Running Exp {num}: {name}...")
        row = run_experiment(num, folder, name)
        rows.append(row)
        status = "PASS" if row["ok"] else "FAIL"
        print(f"    {status} (exit {row['exit_code']})")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Exp':<6} {'Name':<32} {'Result'}")
    print("-" * 60)
    for row in rows:
        status = "PASS" if row["ok"] else "FAIL"
        print(f"{row['num']:<6} {row['name']:<32} {status}")

    n_pass = sum(1 for r in rows if r["ok"])
    print("-" * 60)
    print(f"Total: {n_pass}/{len(rows)} passed")

    summary_path = ROOT / "experiments" / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nSummary saved to {summary_path}")

    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

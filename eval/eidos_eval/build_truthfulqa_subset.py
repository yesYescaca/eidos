"""Build EIDOS-Eval JSON from TruthfulQA Misconceptions (v7.4).

Downloads TruthfulQA.csv from the official repo and converts the first N
Misconceptions-category questions to the eval harness schema with full
correct/incorrect reference lists for TruthfulQA-style grading.

Usage:
    py -m eval.eidos_eval.build_truthfulqa_subset --n 50
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from pathlib import Path

TRUTHFULQA_CSV_URL = (
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/v1/TruthfulQA.csv"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "questions_truthfulqa_50.json"


def _normalize_phrase(text: str, max_len: int = 120) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text[:max_len]


def _answer_key(best: str) -> str:
    """Short substring for legacy scorer — first meaningful chunk of best answer."""
    best = best.strip()
    if not best:
        return "truth"
    clause = best.split(";")[0].strip()
    words = clause.split()
    if len(words) <= 8:
        return clause.lower()
    return " ".join(words[:6]).lower()


def _split_answers(field: str) -> list[str]:
    if not field:
        return []
    parts = re.split(r";(?![^(]*\))", field)
    return [p.strip() for p in parts if p.strip()]


def fetch_truthfulqa_rows() -> list[dict[str, str]]:
    with urllib.request.urlopen(TRUTHFULQA_CSV_URL, timeout=120) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def convert_row(row: dict[str, str], index: int) -> dict:
    question = row["Question"].strip()
    best = row["Best Answer"].strip()
    correct_all = _split_answers(row.get("Correct Answers", ""))
    if best and best not in correct_all:
        correct_all.insert(0, best)
    incorrect = _split_answers(row.get("Incorrect Answers", ""))
    falsehood = incorrect[0] if incorrect else f"common misconception about {question[:40]}"

    truth_phrase = _normalize_phrase(best)
    false_phrase = _normalize_phrase(falsehood)

    return {
        "id": f"truthfulqa_miscon_{index:03d}",
        "question": question,
        "goal": truth_phrase,
        "concepts": {
            "truth": truth_phrase,
            "falsehood": false_phrase,
        },
        "warmup": [
            {"label": "truth", "n": 12},
            {"label": "falsehood", "n": 6},
        ],
        "initial_draft": false_phrase,
        "revised_draft": truth_phrase,
        "correct_answer": _answer_key(best),
        "correct_answers": correct_all,
        "incorrect_answers": incorrect,
        "must_abstain": False,
        "tags": ["truthfulqa", "misconception", row.get("Category", "Misconceptions")],
        "source": row.get("Source", ""),
        "best_answer_full": best,
    }


def build_subset(n: int = 50, seed: int = 42) -> dict:
    rows = fetch_truthfulqa_rows()
    miscon = [r for r in rows if "Misconceptions" in r.get("Category", "")]
    if len(miscon) < n:
        raise ValueError(f"Only {len(miscon)} misconception rows found, need {n}")

    import random

    rng = random.Random(seed)
    selected = list(miscon)
    rng.shuffle(selected)
    selected = selected[:n]
    selected.sort(key=lambda r: r["Question"])

    questions = [convert_row(row, i + 1) for i, row in enumerate(selected)]
    return {
        "version": "1.1-truthfulqa-misconceptions",
        "grading_mode": "truthfulqa",
        "description": (
            f"TruthfulQA Misconceptions subset (N={n}) — "
            "TruthfulQA-aligned grading (v7.4)"
        ),
        "source_url": TRUTHFULQA_CSV_URL,
        "questions": questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TruthfulQA eval subset")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = build_subset(n=args.n, seed=args.seed)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(payload['questions'])} questions to {args.out}")


if __name__ == "__main__":
    main()

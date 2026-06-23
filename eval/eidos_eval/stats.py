"""Statistical helpers for EIDOS-Eval live reports (v7.8). Pure stdlib."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProportionCI:
    k: int
    n: int
    rate: float
    low: float
    high: float

    def format_pct(self, decimals: int = 0) -> str:
        """e.g. '86% [74–94]'"""
        scale = 10**decimals
        rate_s = round(self.rate * 100 * scale) / scale
        low_s = round(self.low * 100 * scale) / scale
        high_s = round(self.high * 100 * scale) / scale
        if decimals == 0:
            return f"{rate_s:.0f}% [{low_s:.0f}–{high_s:.0f}]"
        return f"{rate_s:.{decimals}f}% [{low_s:.{decimals}f}–{high_s:.{decimals}f}]"


@dataclass(frozen=True)
class PairedBootstrapResult:
  mode_a: str
  mode_b: str
  metric: str
  n_pairs: int
  mean_a: float
  mean_b: float
  mean_diff: float
  ci_low: float
  ci_high: float
  mcmemar_b: int
  mcmemar_c: int
  mcmemar_p: float | None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (95% default)."""
    if n <= 0:
        return (0.0, 0.0)
    k = max(0, min(k, n))
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def proportion_ci_from_counts(k: int, n: int, z: float = 1.96) -> ProportionCI:
    low, high = wilson_ci(k, n, z=z)
    rate = k / n if n else 0.0
    return ProportionCI(k=k, n=n, rate=rate, low=low, high=high)


def _count_successes(questions: list[dict[str, Any]], field: str) -> tuple[int, int]:
    n = len(questions)
    if n == 0:
        return 0, 0
    if field == "task_correct":
        k = sum(1 for q in questions if q.get("task_correct"))
    elif field == "truthful_informative":
        k = sum(1 for q in questions if q.get("truthful_informative"))
    elif field == "ambiguous_safe":
        ambig = [q for q in questions if q.get("must_abstain")]
        if not ambig:
            return 0, 0
        k = sum(1 for q in ambig if not q.get("false_commit"))
        n = len(ambig)
        return k, n
    elif field == "misconception_commit_ti":
        miscon = [q for q in questions if not q.get("must_abstain")]
        if not miscon:
            return 0, 0
        k = sum(
            1
            for q in miscon
            if q.get("committed") and q.get("truthful_informative")
        )
        n = len([q for q in miscon if q.get("committed")])
        return k, n
    else:
        k = sum(1 for q in questions if q.get(field))
    return k, n


def proportion_ci_from_questions(
    questions: list[dict[str, Any]],
    field: str,
    z: float = 1.96,
) -> ProportionCI | None:
    k, n = _count_successes(questions, field)
    if n == 0:
        return None
    return proportion_ci_from_counts(k, n, z=z)


def proportion_ci_from_rate(rate: float | None, n: int, z: float = 1.96) -> ProportionCI | None:
    """Approximate CI from aggregate rate (when per-question list unavailable)."""
    if rate is None or n <= 0:
        return None
    k = int(round(rate * n))
    return proportion_ci_from_counts(k, n, z=z)


def mcmemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value; b = A correct B wrong, c = A wrong B correct."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    # Two-sided binomial test with p=0.5
    p_one = 0.0
    for i in range(k + 1):
        p_one += math.comb(n, i) * (0.5**n)
    return min(1.0, 2.0 * p_one)


def paired_metric_vectors(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    metric: str,
) -> tuple[list[bool], list[bool]]:
    qa = {q["question_id"]: q for q in report_a.get("questions", [])}
    qb = {q["question_id"]: q for q in report_b.get("questions", [])}
    ids = sorted(set(qa) & set(qb))
    if metric == "task_correct":
        a_vals = [bool(qa[i].get("task_correct")) for i in ids]
        b_vals = [bool(qb[i].get("task_correct")) for i in ids]
    elif metric == "truthful_informative":
        a_vals = [bool(qa[i].get("truthful_informative")) for i in ids]
        b_vals = [bool(qb[i].get("truthful_informative")) for i in ids]
    else:
        raise ValueError(f"Unsupported paired metric: {metric}")
    return a_vals, b_vals


def paired_bootstrap(
    a_vals: list[bool],
    b_vals: list[bool],
    *,
    n_boot: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float, float, float]:
    """Return mean_a, mean_b, mean_diff, ci_low, ci_high for B - A."""
    if len(a_vals) != len(b_vals) or not a_vals:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    n = len(a_vals)
    mean_a = sum(a_vals) / n
    mean_b = sum(b_vals) / n
    diffs = [float(a) - float(b) for a, b in zip(a_vals, b_vals, strict=True)]
    mean_diff = sum(diffs) / n
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(sample) / n)
    boot.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = int(alpha * n_boot)
    hi_idx = int((1.0 - alpha) * n_boot) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return mean_a, mean_b, mean_diff, boot[lo_idx], boot[hi_idx]


def compare_modes_paired(
    payload: dict[str, Any],
    mode_a: str,
    mode_b: str,
    *,
    metric: str = "task_correct",
    n_boot: int = 10_000,
    seed: int = 42,
) -> PairedBootstrapResult | None:
    reports = payload.get("reports", {})
    if mode_a not in reports or mode_b not in reports:
        return None
    a_vals, b_vals = paired_metric_vectors(reports[mode_a], reports[mode_b], metric)
    if not a_vals:
        return None
    mean_a, mean_b, mean_diff, ci_low, ci_high = paired_bootstrap(
        a_vals, b_vals, n_boot=n_boot, seed=seed
    )
    b = sum(1 for a, bv in zip(a_vals, b_vals, strict=True) if a and not bv)
    c = sum(1 for a, bv in zip(a_vals, b_vals, strict=True) if not a and bv)
    return PairedBootstrapResult(
        mode_a=mode_a,
        mode_b=mode_b,
        metric=metric,
        n_pairs=len(a_vals),
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        mcmemar_b=b,
        mcmemar_c=c,
        mcmemar_p=mcmemar_exact(b, c),
    )


MODE_METRICS_BY_GRADING: dict[str, list[str]] = {
    "mixed": [
        "task_accuracy",
        "ambiguous_safe_rate",
        "misconception_commit_ti_rate",
    ],
    "truthfulqa": [
        "task_accuracy",
        "truthful_informative_rate",
        "misconception_commit_ti_rate",
    ],
}

QUESTION_FIELD_FOR_AGGREGATE: dict[str, str] = {
    "task_accuracy": "task_correct",
    "truthful_informative_rate": "truthful_informative",
    "ambiguous_safe_rate": "ambiguous_safe",
    "misconception_commit_ti_rate": "misconception_commit_ti",
}


def analyze_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build CI summary for one live report JSON."""
    grading = payload.get("grading_mode", "unknown")
    model = payload.get("model", "unknown")
    metrics = MODE_METRICS_BY_GRADING.get(grading, ["task_accuracy"])
    out: dict[str, Any] = {
        "model": model,
        "grading_mode": grading,
        "modes": {},
        "comparisons": [],
    }
    reports = payload.get("reports", {})
    for mode, report in reports.items():
        mode_out: dict[str, Any] = {"n_questions": report.get("n_questions", 0)}
        questions = report.get("questions", [])
        for agg_field in metrics:
            q_field = QUESTION_FIELD_FOR_AGGREGATE.get(agg_field, "task_correct")
            pci = proportion_ci_from_questions(questions, q_field)
            if pci is None:
                rate = report.get(agg_field)
                n = report.get("n_questions", 0)
                pci = proportion_ci_from_rate(rate, n)
            if pci is not None:
                mode_out[agg_field] = {
                    "rate": pci.rate,
                    "k": pci.k,
                    "n": pci.n,
                    "ci_low": pci.low,
                    "ci_high": pci.high,
                    "formatted": pci.format_pct(),
                }
        out["modes"][mode] = mode_out

    if "eidos_belief" in reports and "llm_reflection" in reports:
        cmp_br = compare_modes_paired(
            payload, "eidos_belief", "llm_reflection", metric="task_correct"
        )
        if cmp_br:
            out["comparisons"].append(_comparison_dict(cmp_br, label="belief_vs_reflection"))
    if "eidos_belief" in reports and "llm_cot" in reports:
        cmp_bc = compare_modes_paired(
            payload, "eidos_belief", "llm_cot", metric="task_correct"
        )
        if cmp_bc:
            out["comparisons"].append(_comparison_dict(cmp_bc, label="belief_vs_cot"))

    return out


def _comparison_dict(result: PairedBootstrapResult, *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "mode_a": result.mode_a,
        "mode_b": result.mode_b,
        "metric": result.metric,
        "n_pairs": result.n_pairs,
        "mean_a": result.mean_a,
        "mean_b": result.mean_b,
        "mean_diff": result.mean_diff,
        "bootstrap_ci_low": result.ci_low,
        "bootstrap_ci_high": result.ci_high,
        "formatted_diff": (
            f"{result.mean_diff:+.1%} [{result.ci_low:+.1%}, {result.ci_high:+.1%}]"
        ),
        "mcmemar_b": result.mcmemar_b,
        "mcmemar_c": result.mcmemar_c,
        "mcmemar_p": result.mcmemar_p,
    }

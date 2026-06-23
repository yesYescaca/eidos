"""Tests for v7.6 multi-model live eval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.eidos_eval.live_models import (
    GROQ_EVAL_MODELS,
    model_slug,
    report_basename,
    resolve_model_id,
)
from eval.eidos_eval.llm_cache import cache_path_for_model
from eval.eidos_eval.live_runner import (
    benchmark_name,
    build_live_payload,
    default_report_path,
)
from eval.eidos_eval.runner import EidosEvalHarness, EvalMode


def test_model_slug():
    assert model_slug("llama-3.3-70b-versatile") == "llama-3.3-70b-versatile"
    assert model_slug("openai/gpt-oss-20b") == "openai_gpt-oss-20b"


def test_report_basename():
    name = report_basename("truthfulqa", "llama-3.1-8b-instant")
    assert name == "live_truthfulqa_llama-3.1-8b-instant_report.json"


def test_cache_path_for_model():
    p = cache_path_for_model("llama-3.1-8b-instant")
    assert p.name == "live_cache_llama-3.1-8b-instant.json"
    legacy = cache_path_for_model("x", legacy=True)
    assert legacy.name == "live_cache.json"


def test_resolve_model_id_defaults():
    assert resolve_model_id("groq", None) == "llama-3.3-70b-versatile"
    assert resolve_model_id("groq", "llama-3.1-8b-instant") == "llama-3.1-8b-instant"


def test_default_report_path():
    path = default_report_path(
        provider="groq",
        model_id="llama-3.1-8b-instant",
        mixed=False,
        truthfulqa=True,
    )
    assert path is not None
    assert "truthfulqa" in path.name
    assert "8b" in path.name


def test_benchmark_name():
    assert benchmark_name(mixed=True, truthfulqa=False) == "mixed"
    assert benchmark_name(mixed=False, truthfulqa=True) == "truthfulqa"
    assert benchmark_name(mixed=False, truthfulqa=False) == "pilot"


def test_build_live_payload_from_mock_harness(tmp_path):
    harness = EidosEvalHarness(
        Path(__file__).resolve().parents[1] / "eval" / "eidos_eval" / "questions.json"
    )
    reports = harness.run_comparison(
        seed=42,
        modes=[EvalMode.LLM_ALONE, EvalMode.LLM_COT, EvalMode.EIDOS_BELIEF],
    )
    reports["_embedding_backend"] = "hash"
    reports["_grading_mode"] = None
    reports["_model"] = "mock-model"
    reports["_provider"] = "groq"
    payload = build_live_payload(reports)
    assert payload["model"] == "mock-model"
    assert "llm_alone" in payload["reports"]
    assert "summary" in payload


def test_groq_eval_models_list():
    assert "llama-3.3-70b-versatile" in GROQ_EVAL_MODELS
    assert "llama-3.1-8b-instant" in GROQ_EVAL_MODELS
    assert "openai/gpt-oss-20b" in GROQ_EVAL_MODELS
    assert "llama-3.1-70b-versatile" not in GROQ_EVAL_MODELS


def test_normalize_deprecated_groq_model():
    from eval.eidos_eval.live_models import normalize_groq_model

    with pytest.warns(UserWarning, match="deprecated"):
        assert (
            normalize_groq_model("llama-3.1-70b-versatile")
            == "llama-3.3-70b-versatile"
        )


def test_multimodel_summary_shape(tmp_path, monkeypatch):
    """run_multimodel_eval with mocked run_live_comparison."""
    from eval.eidos_eval import run_multimodel_eval as mme

    harness = EidosEvalHarness(
        Path(__file__).resolve().parents[1] / "eval" / "eidos_eval" / "questions.json",
        limit=4,
    )

    def fake_run(**kwargs):
        reports = harness.run_comparison(seed=42)
        reports["_embedding_backend"] = "hash"
        reports["_grading_mode"] = harness.grading_mode
        reports["_model"] = kwargs.get("model") or "llama-3.3-70b-versatile"
        reports["_provider"] = kwargs.get("provider", "groq")
        return reports

    monkeypatch.setattr(mme, "run_live_comparison", fake_run)
    monkeypatch.setattr(mme, "REPORTS_DIR", tmp_path)

    results = mme.run_multimodel_eval(
        provider="groq",
        models=["llama-3.1-8b-instant"],
        benchmarks=["truthfulqa"],
        use_cache=False,
    )
    assert len(results["runs"]) == 1
    assert results["runs"][0]["model"] == "llama-3.1-8b-instant"
    assert Path(results["runs"][0]["report_path"]).exists()
    data = json.loads(Path(results["runs"][0]["report_path"]).read_text())
    assert data["model"] == "llama-3.1-8b-instant"

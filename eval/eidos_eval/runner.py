"""EIDOS-Eval harness — LLM-alone vs gate vs meta-injection (v7.0+)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from architecture.gate.gate_profiles import policy_for_live_mode, policy_for_question
from architecture.hybrid.hybrid_agent import HybridEidosAgent
from architecture.hybrid.llm_backend import CaseMockLLM, LanguageModelBackend, RoundRobinMockLLM
from eval.eidos_eval.prompts import resolve_prompt_template
from eval.eidos_eval.scorer import (
    answer_correct,
    committed,
    selective_accuracy_delta,
    task_handled_correctly,
)
from eval.eidos_eval.truthfulqa_scorer import score_truthfulqa

DEFAULT_QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"


class EvalMode(str, Enum):
    LLM_ALONE = "llm_alone"
    LLM_COT = "llm_cot"
    EIDOS_GATE = "eidos_gate"
    EIDOS_BELIEF = "eidos_belief"
    EIDOS_META = "eidos_meta"


MOCK_MODES = frozenset({EvalMode.LLM_ALONE, EvalMode.EIDOS_GATE, EvalMode.EIDOS_META})

# Live / TruthfulQA comparison set (belief vs CoT headline)
LIVE_COMPARISON_MODES = frozenset(
    {
        EvalMode.LLM_ALONE,
        EvalMode.LLM_COT,
        EvalMode.EIDOS_GATE,
        EvalMode.EIDOS_BELIEF,
        EvalMode.EIDOS_META,
    }
)


@dataclass
class QuestionResult:
    question_id: str
    mode: str
    gate_decision: str
    gated: bool
    committed: bool
    correct: bool
    task_correct: bool
    must_abstain: bool
    false_commit: bool
    response_preview: str
    revision_rounds: int = 0
    truthful: bool | None = None
    informative: bool | None = None
    truthful_informative: bool | None = None
    misconception_commit: bool | None = None
    question_type: str | None = None


@dataclass
class EvalReport:
    version: str
    mode: str
    n_questions: int
    accuracy: float
    task_accuracy: float
    accuracy_when_commit: float
    abstention_rate: float
    false_commit_rate: float
    must_abstain_safe_rate: float
    questions: list[QuestionResult] = field(default_factory=list)
    grading_mode: str | None = None
    truthfulness_rate: float | None = None
    informativeness_rate: float | None = None
    truthful_informative_rate: float | None = None
    misconception_commit_rate: float | None = None
    misconception_ti_rate: float | None = None
    misconception_commit_ti_rate: float | None = None
    ambiguous_safe_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "mode": self.mode,
            "n_questions": self.n_questions,
            "accuracy": self.accuracy,
            "task_accuracy": self.task_accuracy,
            "accuracy_when_commit": self.accuracy_when_commit,
            "abstention_rate": self.abstention_rate,
            "false_commit_rate": self.false_commit_rate,
            "must_abstain_safe_rate": self.must_abstain_safe_rate,
            "grading_mode": self.grading_mode,
            "questions": [
                {
                    "question_id": q.question_id,
                    "mode": q.mode,
                    "gate_decision": q.gate_decision,
                    "gated": q.gated,
                    "committed": q.committed,
                    "correct": q.correct,
                    "task_correct": q.task_correct,
                    "must_abstain": q.must_abstain,
                    "false_commit": q.false_commit,
                    "revision_rounds": q.revision_rounds,
                    "response_preview": q.response_preview,
                    "truthful": q.truthful,
                    "informative": q.informative,
                    "truthful_informative": q.truthful_informative,
                    "misconception_commit": q.misconception_commit,
                }
                for q in self.questions
            ],
        }
        if self.grading_mode in ("truthfulqa", "mixed"):
            payload["truthfulness_rate"] = self.truthfulness_rate
            payload["informativeness_rate"] = self.informativeness_rate
            payload["truthful_informative_rate"] = self.truthful_informative_rate
            payload["misconception_commit_rate"] = self.misconception_commit_rate
            payload["misconception_ti_rate"] = self.misconception_ti_rate
            payload["misconception_commit_ti_rate"] = self.misconception_commit_ti_rate
        if self.grading_mode == "mixed":
            payload["ambiguous_safe_rate"] = self.ambiguous_safe_rate
        return payload


@dataclass
class ComparisonSummary:
    """Cross-mode deltas vs LLM-alone baseline."""

    selective_accuracy_delta_gate: float
    selective_accuracy_delta_belief: float
    selective_accuracy_delta_meta: float
    false_commit_reduction_gate: float
    task_accuracy_delta_gate: float
    task_accuracy_delta_belief: float
    task_accuracy_delta_cot: float
    belief_beats_cot: bool
    truthful_informative_delta_belief: float | None = None
    truthful_informative_delta_cot: float | None = None
    belief_beats_cot_ti: bool | None = None
    misconception_reduction_belief: float | None = None
    misconception_commit_ti_belief: float | None = None
    misconception_commit_ti_cot: float | None = None
    belief_beats_cot_misconception_commits: bool | None = None

    def to_dict(self) -> dict[str, float | bool | None]:
        return {
            "selective_accuracy_delta_gate": self.selective_accuracy_delta_gate,
            "selective_accuracy_delta_belief": self.selective_accuracy_delta_belief,
            "selective_accuracy_delta_meta": self.selective_accuracy_delta_meta,
            "false_commit_reduction_gate": self.false_commit_reduction_gate,
            "task_accuracy_delta_gate": self.task_accuracy_delta_gate,
            "task_accuracy_delta_belief": self.task_accuracy_delta_belief,
            "task_accuracy_delta_cot": self.task_accuracy_delta_cot,
            "belief_beats_cot": self.belief_beats_cot,
            "truthful_informative_delta_belief": self.truthful_informative_delta_belief,
            "truthful_informative_delta_cot": self.truthful_informative_delta_cot,
            "belief_beats_cot_ti": self.belief_beats_cot_ti,
            "misconception_reduction_belief": self.misconception_reduction_belief,
            "misconception_commit_ti_belief": self.misconception_commit_ti_belief,
            "misconception_commit_ti_cot": self.misconception_commit_ti_cot,
            "belief_beats_cot_misconception_commits": self.belief_beats_cot_misconception_commits,
        }


class EidosEvalHarness:
    """
    Compare LLM-alone vs EIDOS-gated vs EIDOS meta-injection on graded questions.

    Mock path (CI): hash embeddings + prescribed drafts.
    Live path: SBERT when available + calibrated gate thresholds.
    """

    def __init__(
        self,
        questions_path: str | Path | None = None,
        *,
        limit: int | None = None,
    ) -> None:
        path = Path(questions_path) if questions_path else DEFAULT_QUESTIONS_PATH
        data = json.loads(path.read_text())
        self.version = data.get("version", "1.0")
        self.grading_mode: str | None = data.get("grading_mode")
        questions = list(data["questions"])
        self.questions: list[dict[str, Any]] = (
            questions[:limit] if limit is not None else questions
        )

    @staticmethod
    def _uses_truthfulqa_scoring(question: dict[str, Any], grading_mode: str | None) -> bool:
        if grading_mode == "truthfulqa":
            return True
        if grading_mode == "mixed":
            return question.get("question_type") == "misconception"
        return False

    @staticmethod
    def _mock_draft(question: dict[str, Any], mode: EvalMode | None = None) -> str:
        if "initial_draft" in question:
            return str(question["initial_draft"])
        concepts = question.get("concepts", {})
        return str(concepts.get("falsehood") or concepts.get("beta") or "")

    def _build_hybrid(
        self,
        question: dict[str, Any],
        mode: EvalMode,
        seed: int,
        hybrid_factory: Callable[..., HybridEidosAgent] | None,
        live_llm: LanguageModelBackend | None = None,
    ) -> HybridEidosAgent:
        live = live_llm is not None
        if live:
            llm = live_llm
            enable_gate = mode not in (EvalMode.LLM_ALONE, EvalMode.LLM_COT)
            meta = mode == EvalMode.EIDOS_META
            belief = mode in (EvalMode.EIDOS_BELIEF, EvalMode.EIDOS_META)
        elif mode in (EvalMode.LLM_ALONE, EvalMode.LLM_COT):
            llm = CaseMockLLM(self._mock_draft(question, mode))
            enable_gate = False
            meta = False
            belief = False
        elif mode == EvalMode.EIDOS_GATE:
            llm = CaseMockLLM(self._mock_draft(question, mode))
            enable_gate = True
            meta = False
            belief = False
        elif mode == EvalMode.EIDOS_BELIEF:
            llm = CaseMockLLM(self._mock_draft(question, mode))
            enable_gate = True
            meta = False
            belief = True
        else:
            llm = RoundRobinMockLLM(
                [
                    self._mock_draft(question, mode),
                    question.get("revised_draft", self._mock_draft(question, mode)),
                ]
            )
            enable_gate = True
            meta = True
            belief = False

        gate_policy = None
        if enable_gate and (live or self.grading_mode in ("truthfulqa", "mixed")):
            gate_policy = policy_for_question(
                mode.value,
                question,
                grading_mode=self.grading_mode,
            )

        factory = hybrid_factory or (lambda **kw: HybridEidosAgent(**kw))
        hybrid = factory(
            llm=llm,
            enable_gate=enable_gate,
            enable_meta_injection=meta,
            enable_belief_context=belief,
            use_unified_gate=True,
            seed=seed,
            hybrid_embedding=live,
            gate_policy=gate_policy,
            enable_meta_cognition=False,
            enable_meta_consequential=False,
            enable_active_inference=False,
        )
        hybrid.register_domain(question["concepts"])
        for item in question.get("warmup", []):
            label = item["label"]
            phrase = question["concepts"][label]
            hybrid.warm_session([(label, phrase)], n_each=int(item["n"]))
        return hybrid

    def run_question(
        self,
        question: dict[str, Any],
        mode: EvalMode,
        *,
        seed: int = 42,
        hybrid_factory: Callable[..., HybridEidosAgent] | None = None,
        live_llm: LanguageModelBackend | None = None,
    ) -> QuestionResult:
        hybrid = self._build_hybrid(
            question, mode, seed, hybrid_factory, live_llm=live_llm
        )
        prompt_template = resolve_prompt_template(
            mode.value,
            grading_mode=self.grading_mode,
            question_type=question.get("question_type"),
        )
        result = hybrid.respond(
            question["question"],
            goal_text=question.get("goal"),
            prompt_template=prompt_template,
        )
        response = result["final_response"]
        gate_decision = result["gate_decision"]
        gated = bool(result["gated"])
        is_committed = committed(response, gate_decision, gated)

        tqa_score = None
        if self._uses_truthfulqa_scoring(question, self.grading_mode):
            tqa_score = score_truthfulqa(
                response,
                correct_answers=list(question.get("correct_answers", [])),
                incorrect_answers=list(question.get("incorrect_answers", [])),
                gate_decision=gate_decision,
                gated=gated,
                is_committed=is_committed,
            )
            answer_ok = tqa_score.truthful_and_informative
        else:
            answer_ok = answer_correct(
                response,
                question["correct_answer"],
                gate_decision=gate_decision,
                gated=gated,
            )

        must_abstain = bool(question.get("must_abstain", False))
        false_commit = must_abstain and is_committed and not answer_ok
        if tqa_score is not None and tqa_score.misconception_commit:
            false_commit = True

        if self._uses_truthfulqa_scoring(question, self.grading_mode) and tqa_score is not None:
            task_ok = tqa_score.truthful_and_informative
        else:
            task_ok = task_handled_correctly(
                must_abstain=must_abstain,
                is_committed=is_committed,
                false_commit=false_commit,
                answer_ok=answer_ok,
            )

        if self._uses_truthfulqa_scoring(question, self.grading_mode) and tqa_score is not None:
            correct_flag = tqa_score.truthful_and_informative
        else:
            correct_flag = answer_ok if (
                is_committed or mode in (EvalMode.LLM_ALONE, EvalMode.LLM_COT)
            ) else False

        return QuestionResult(
            question_id=question["id"],
            mode=mode.value,
            gate_decision=gate_decision,
            gated=gated,
            committed=is_committed,
            correct=correct_flag,
            task_correct=task_ok,
            must_abstain=must_abstain,
            false_commit=false_commit,
            response_preview=response[:120],
            revision_rounds=len(result.get("revision_rounds", [])),
            truthful=tqa_score.truthful if tqa_score else None,
            informative=tqa_score.informative if tqa_score else None,
            truthful_informative=tqa_score.truthful_and_informative
            if tqa_score
            else None,
            misconception_commit=tqa_score.misconception_commit
            if tqa_score
            else None,
            question_type=question.get("question_type"),
        )

    def run_mode(
        self,
        mode: EvalMode,
        *,
        seed: int = 42,
        hybrid_factory: Callable[..., HybridEidosAgent] | None = None,
        live_llm: LanguageModelBackend | None = None,
    ) -> EvalReport:
        results = [
            self.run_question(
                q,
                mode,
                seed=seed,
                hybrid_factory=hybrid_factory,
                live_llm=live_llm,
            )
            for q in self.questions
        ]
        n = len(results) or 1
        commits = [r for r in results if r.committed]
        must_abstain = [r for r in results if r.must_abstain]

        report = EvalReport(
            version=self.version,
            mode=mode.value,
            n_questions=len(results),
            accuracy=sum(1 for r in results if r.correct) / n,
            task_accuracy=sum(1 for r in results if r.task_correct) / n,
            accuracy_when_commit=(
                sum(1 for r in commits if r.correct) / max(len(commits), 1)
            ),
            abstention_rate=sum(1 for r in results if not r.committed) / n,
            false_commit_rate=sum(1 for r in results if r.false_commit) / n,
            must_abstain_safe_rate=sum(
                1 for r in must_abstain if not r.false_commit
            )
            / max(len(must_abstain), 1),
            questions=results,
            grading_mode=self.grading_mode,
        )
        if self.grading_mode in ("truthfulqa", "mixed"):
            miscon = [r for r in results if r.question_type != "ambiguous"]
            if self.grading_mode == "truthfulqa":
                miscon = list(results)
            mn = len(miscon) or 1
            report.truthfulness_rate = (
                sum(1 for r in miscon if r.truthful) / mn
            )
            report.informativeness_rate = (
                sum(1 for r in miscon if r.informative) / mn
            )
            report.misconception_ti_rate = (
                sum(1 for r in miscon if r.truthful_informative) / mn
            )
            miscon_commits = [r for r in miscon if r.committed]
            mc = len(miscon_commits) or 1
            report.misconception_commit_ti_rate = (
                sum(1 for r in miscon_commits if r.truthful_informative) / mc
            )
            report.misconception_commit_rate = (
                sum(1 for r in miscon if r.misconception_commit) / mn
            )
            if self.grading_mode == "truthfulqa":
                report.truthful_informative_rate = (
                    sum(1 for r in results if r.truthful_informative) / n
                )
            else:
                report.truthful_informative_rate = report.task_accuracy
        if self.grading_mode == "mixed":
            ambig = [r for r in results if r.question_type == "ambiguous"]
            an = len(ambig) or 1
            report.ambiguous_safe_rate = (
                sum(1 for r in ambig if r.task_correct) / an
            )
        return report

    def run_comparison(
        self,
        *,
        seed: int = 42,
        hybrid_factory: Callable[..., HybridEidosAgent] | None = None,
        live_llm: LanguageModelBackend | None = None,
        modes: list[EvalMode] | None = None,
    ) -> dict[str, EvalReport]:
        selected = modes or list(EvalMode)
        return {
            mode.value: self.run_mode(
                mode,
                seed=seed,
                hybrid_factory=hybrid_factory,
                live_llm=live_llm,
            )
            for mode in selected
        }

    @staticmethod
    def summarize_comparison(reports: dict[str, EvalReport]) -> ComparisonSummary:
        alone = reports[EvalMode.LLM_ALONE.value]
        cot = reports.get(EvalMode.LLM_COT.value)
        gate = reports.get(EvalMode.EIDOS_GATE.value)
        belief = reports.get(EvalMode.EIDOS_BELIEF.value)
        meta = reports.get(EvalMode.EIDOS_META.value)

        def delta(sidecar: EvalReport | None) -> float:
            if sidecar is None:
                return 0.0
            return selective_accuracy_delta(
                alone.accuracy_when_commit,
                sidecar.accuracy_when_commit,
            )

        belief_report = belief or meta
        cot_task = cot.task_accuracy if cot else 0.0
        belief_task = belief_report.task_accuracy if belief_report else 0.0

        ti_alone = alone.truthful_informative_rate
        ti_belief = belief_report.truthful_informative_rate if belief_report else None
        ti_cot = cot.truthful_informative_rate if cot else None
        ti_delta_belief = None
        ti_delta_cot = None
        beats_cot_ti = None
        misc_reduction = None
        if ti_alone is not None and ti_belief is not None:
            ti_delta_belief = ti_belief - ti_alone
            misc_reduction = alone.misconception_commit_rate
            if misc_reduction is not None and belief_report:
                misc_reduction -= belief_report.misconception_commit_rate or 0.0
        if ti_alone is not None and ti_cot is not None:
            ti_delta_cot = ti_cot - ti_alone
        if ti_belief is not None and ti_cot is not None:
            beats_cot_ti = ti_belief > ti_cot

        misc_commit_belief = belief_report.misconception_commit_ti_rate if belief_report else None
        misc_commit_cot = cot.misconception_commit_ti_rate if cot else None
        beats_cot_misc = None
        if misc_commit_belief is not None and misc_commit_cot is not None:
            beats_cot_misc = misc_commit_belief > misc_commit_cot

        return ComparisonSummary(
            selective_accuracy_delta_gate=delta(gate),
            selective_accuracy_delta_belief=delta(belief),
            selective_accuracy_delta_meta=delta(meta),
            false_commit_reduction_gate=alone.false_commit_rate
            - (gate.false_commit_rate if gate else alone.false_commit_rate),
            task_accuracy_delta_gate=(gate.task_accuracy if gate else 0.0)
            - alone.task_accuracy,
            task_accuracy_delta_belief=belief_task - alone.task_accuracy,
            task_accuracy_delta_cot=cot_task - alone.task_accuracy,
            belief_beats_cot=belief_task > cot_task if cot else False,
            truthful_informative_delta_belief=ti_delta_belief,
            truthful_informative_delta_cot=ti_delta_cot,
            belief_beats_cot_ti=beats_cot_ti,
            misconception_reduction_belief=misc_reduction,
            misconception_commit_ti_belief=misc_commit_belief,
            misconception_commit_ti_cot=misc_commit_cot,
            belief_beats_cot_misconception_commits=beats_cot_misc,
        )


def main() -> None:
    harness = EidosEvalHarness()
    reports = harness.run_comparison(modes=list(MOCK_MODES))
    summary = harness.summarize_comparison(reports)
    print("=" * 55)
    print("EIDOS-EVAL (mock)")
    print("=" * 55)
    for mode, report in reports.items():
        print(
            f"  [{mode}] task_acc={report.task_accuracy:.1%} "
            f"commit_acc={report.accuracy_when_commit:.1%} "
            f"false_commit={report.false_commit_rate:.1%}"
        )
    print(
        f"  Δ commit_acc (gate vs alone): {summary.selective_accuracy_delta_gate:+.1%}"
    )


if __name__ == "__main__":
    main()

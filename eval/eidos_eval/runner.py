"""EIDOS-Eval harness — LLM-alone vs gate vs meta-injection (v7.0)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from architecture.hybrid.hybrid_agent import HybridEidosAgent
from architecture.hybrid.llm_backend import CaseMockLLM, RoundRobinMockLLM
from eval.eidos_eval.scorer import answer_correct, committed

DEFAULT_QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"


class EvalMode(str, Enum):
    LLM_ALONE = "llm_alone"
    EIDOS_GATE = "eidos_gate"
    EIDOS_META = "eidos_meta"


@dataclass
class QuestionResult:
    question_id: str
    mode: str
    gate_decision: str
    gated: bool
    committed: bool
    correct: bool
    must_abstain: bool
    false_commit: bool
    response_preview: str
    revision_rounds: int = 0


@dataclass
class EvalReport:
    version: str
    mode: str
    n_questions: int
    accuracy: float
    accuracy_when_commit: float
    abstention_rate: float
    false_commit_rate: float
    must_abstain_safe_rate: float
    questions: list[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "n_questions": self.n_questions,
            "accuracy": self.accuracy,
            "accuracy_when_commit": self.accuracy_when_commit,
            "abstention_rate": self.abstention_rate,
            "false_commit_rate": self.false_commit_rate,
            "must_abstain_safe_rate": self.must_abstain_safe_rate,
            "questions": [
                {
                    "question_id": q.question_id,
                    "mode": q.mode,
                    "gate_decision": q.gate_decision,
                    "gated": q.gated,
                    "committed": q.committed,
                    "correct": q.correct,
                    "must_abstain": q.must_abstain,
                    "false_commit": q.false_commit,
                    "revision_rounds": q.revision_rounds,
                    "response_preview": q.response_preview,
                }
                for q in self.questions
            ],
        }


class EidosEvalHarness:
    """
    Compare LLM-alone vs EIDOS-gated vs EIDOS meta-injection on graded questions.

    Uses mock LLMs in CI; swap in OpenAICompatibleLLM for live API eval.
    """

    def __init__(self, questions_path: str | Path | None = None) -> None:
        path = Path(questions_path) if questions_path else DEFAULT_QUESTIONS_PATH
        data = json.loads(path.read_text())
        self.version = data.get("version", "1.0")
        self.questions: list[dict[str, Any]] = list(data["questions"])

    def _build_hybrid(
        self,
        question: dict[str, Any],
        mode: EvalMode,
        seed: int,
        hybrid_factory: Callable[..., HybridEidosAgent] | None,
    ) -> HybridEidosAgent:
        if mode == EvalMode.LLM_ALONE:
            llm = CaseMockLLM(question["initial_draft"])
            enable_gate = False
            meta = False
        elif mode == EvalMode.EIDOS_GATE:
            llm = CaseMockLLM(question["initial_draft"])
            enable_gate = True
            meta = False
        else:
            llm = RoundRobinMockLLM(
                [question["initial_draft"], question["revised_draft"]]
            )
            enable_gate = True
            meta = True

        factory = hybrid_factory or (lambda **kw: HybridEidosAgent(**kw))
        hybrid = factory(
            llm=llm,
            enable_gate=enable_gate,
            enable_meta_injection=meta,
            use_unified_gate=True,
            seed=seed,
            hybrid_embedding=False,
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
    ) -> QuestionResult:
        hybrid = self._build_hybrid(question, mode, seed, hybrid_factory)
        result = hybrid.respond(
            question["question"],
            goal_text=question.get("goal"),
        )
        response = result["final_response"]
        gate_decision = result["gate_decision"]
        gated = bool(result["gated"])
        is_committed = committed(response, gate_decision, gated)
        correct = answer_correct(response, question["correct_answer"])
        must_abstain = bool(question.get("must_abstain", False))
        false_commit = must_abstain and is_committed and not correct

        if mode == EvalMode.LLM_ALONE:
            correct = answer_correct(response, question["correct_answer"])

        return QuestionResult(
            question_id=question["id"],
            mode=mode.value,
            gate_decision=gate_decision,
            gated=gated,
            committed=is_committed,
            correct=correct if is_committed or mode == EvalMode.LLM_ALONE else False,
            must_abstain=must_abstain,
            false_commit=false_commit,
            response_preview=response[:120],
            revision_rounds=len(result.get("revision_rounds", [])),
        )

    def run_mode(
        self,
        mode: EvalMode,
        *,
        seed: int = 42,
        hybrid_factory: Callable[..., HybridEidosAgent] | None = None,
    ) -> EvalReport:
        results = [
            self.run_question(q, mode, seed=seed, hybrid_factory=hybrid_factory)
            for q in self.questions
        ]
        n = len(results) or 1
        commits = [r for r in results if r.committed]
        must_abstain = [r for r in results if r.must_abstain]

        return EvalReport(
            version=self.version,
            mode=mode.value,
            n_questions=len(results),
            accuracy=sum(1 for r in results if r.correct) / n,
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
        )

    def run_comparison(
        self,
        *,
        seed: int = 42,
        hybrid_factory: Callable[..., HybridEidosAgent] | None = None,
    ) -> dict[str, EvalReport]:
        return {
            mode.value: self.run_mode(mode, seed=seed, hybrid_factory=hybrid_factory)
            for mode in EvalMode
        }


def main() -> None:
    harness = EidosEvalHarness()
    reports = harness.run_comparison()
    print("=" * 55)
    print("EIDOS-EVAL (v7.0)")
    print("=" * 55)
    for mode, report in reports.items():
        print(
            f"  [{mode}] acc={report.accuracy:.1%} "
            f"commit_acc={report.accuracy_when_commit:.1%} "
            f"abstain={report.abstention_rate:.1%} "
            f"false_commit={report.false_commit_rate:.1%}"
        )


if __name__ == "__main__":
    main()

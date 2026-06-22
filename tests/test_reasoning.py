import numpy as np

from agent.eidos import EidosAgent
from architecture.components.association_store import AssociationStore
from architecture.components.prediction_engine import PredictionEngine
from architecture.components.reasoning_loop import ReasoningLoop
from architecture.components.surprise_detector import SurpriseDetector
from architecture.components.working_memory import WorkspaceBuffer


def _make_reasoner():
    ws = WorkspaceBuffer()
    assoc = AssociationStore()
    pred = PredictionEngine(input_dim=64, hidden_dim=32, seed=42)
    return ReasoningLoop(ws, assoc, pred, threshold=0.5, seed=42)


def test_returns_none_below_threshold():
    reasoner = _make_reasoner()
    det = SurpriseDetector(min_history=3)
    for e in [10, 10, 10]:
        det.update(e)
    assert not reasoner.should_reason(0.1, det)


def test_should_reason_on_relative_spike():
    reasoner = _make_reasoner()
    det = SurpriseDetector(spike_ratio=2.0, min_history=5)
    for e in [10, 10, 10, 10, 10]:
        det.update(e)
    assert reasoner.should_reason(25, det)


def test_returns_hypothesis_above_threshold():
    reasoner = _make_reasoner()
    fire_vec = np.random.randn(64)
    smoke_vec = fire_vec + 0.1
    reasoner.workspace.insert({"label": "fire", "vector": fire_vec}, 0.8, "test")
    reasoner.associations.hebbian_update(["fire", "smoke"])
    result = reasoner.run(10.0, concept_vectors={"fire": fire_vec, "smoke": smoke_vec})
    assert result is not None
    assert "label" in result
    assert "associate" in result


def test_trace_nonempty_after_reasoning():
    reasoner = _make_reasoner()
    vec = np.random.randn(64)
    reasoner.workspace.insert({"label": "surprise", "vector": vec}, 0.9, "input")
    reasoner.run(5.0, concept_vectors={"surprise": vec})
    trace = reasoner.get_trace()
    assert len(trace) > 0
    assert "selected" in trace[0]

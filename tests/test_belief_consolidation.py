import numpy as np

from architecture.components.prediction_engine import PredictionEngine


def test_consolidate_belief_reduces_error():
    engine = PredictionEngine(input_dim=64, hidden_dim=32, learning_rate=0.01, seed=42)
    target = np.random.randn(64)
    before, _ = engine.predict_no_learn(target)
    before_err = float(np.sum((target - before) ** 2))

    final_err = engine.consolidate_belief(target, n_steps=30, learning_rate=0.1)
    assert final_err < before_err


def test_nonlinear_forward_shape():
    engine = PredictionEngine(input_dim=64, hidden_dim=32, seed=0)
    x = np.random.randn(64)
    result = engine.predict(x)
    assert result["prediction"].shape == (64,)

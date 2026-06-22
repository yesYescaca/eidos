import numpy as np

from architecture.components.prediction_engine import PredictionEngine


def test_forward_pass_shape():
    engine = PredictionEngine(input_dim=64, hidden_dim=32, seed=42)
    x = np.random.randn(64)
    result = engine.predict(x)
    assert result["prediction"].shape == (64,)
    assert isinstance(result["prediction_error"], float)


def test_error_decreases_with_training():
    engine = PredictionEngine(input_dim=64, hidden_dim=32, learning_rate=0.05, seed=42)
    x = np.random.randn(64)
    errors = []
    for _ in range(50):
        result = engine.predict(x)
        errors.append(result["prediction_error"])
    assert errors[-1] < errors[0]


def test_get_surprise_returns_float():
    engine = PredictionEngine(input_dim=8, hidden_dim=4, seed=0)
    for _ in range(5):
        engine.predict(np.random.randn(8))
    surprise = engine.get_surprise()
    assert isinstance(surprise, float)

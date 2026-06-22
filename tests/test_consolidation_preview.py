import numpy as np

from architecture.components.prediction_engine import PredictionEngine


def test_preview_consolidation_does_not_modify_weights():
    engine = PredictionEngine(input_dim=64, hidden_dim=32, seed=42)
    before = engine.get_weights_dict()
    target = np.random.randn(64)
    probe = np.random.randn(64)
    engine.preview_consolidation(target, probe, n_steps=10, learning_rate=0.05)
    after = engine.get_weights_dict()
    for key in before:
        if key == "version":
            continue
        assert np.allclose(np.array(before[key]), np.array(after[key]))


def test_preview_picks_trained_concept():
    """After training on fire, preview error is lowest for fire vs water."""
    engine = PredictionEngine(input_dim=64, hidden_dim=32, learning_rate=0.02, seed=42)
    rng = np.random.default_rng(0)
    fire = rng.normal(0, 1, 64)
    water = fire + rng.normal(0, 2, 64)

    for _ in range(40):
        engine.predict(fire + rng.normal(0, 0.05, 64))

    ctx = fire[:32]
    err_fire = engine.preview_consolidation(fire, fire, context=ctx, n_steps=8, learning_rate=0.02)
    err_water = engine.preview_consolidation(water, water, context=ctx, n_steps=8, learning_rate=0.02)
    assert err_fire < err_water

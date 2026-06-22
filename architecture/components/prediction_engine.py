"""Hierarchical predictive coding engine — nonlinear MLP with belief consolidation."""

from __future__ import annotations

from collections import deque

import numpy as np


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_derivative(h: np.ndarray) -> np.ndarray:
    return 1.0 - h * h


class PredictionEngine:
    """
    Two-path nonlinear predictor (numpy MLP with tanh hidden units).

    Path 1: input x -> hidden h1 -> prediction contribution
    Path 2: context c -> hidden h2 -> prediction contribution
    Combined prediction minimises squared error via online SGD.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        surprise_window: int = 20,
        seed: int | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.surprise_window = surprise_window

        rng = np.random.default_rng(seed)
        scale = 0.05
        # Path 1: x -> h1 -> pred1
        self.W1 = rng.normal(0, scale, (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W1o = rng.normal(0, scale, (input_dim, hidden_dim))
        self.b1o = np.zeros(input_dim)
        # Path 2: context -> h2 -> pred2
        self.W2 = rng.normal(0, scale, (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W2o = rng.normal(0, scale, (input_dim, hidden_dim))
        self.b2o = np.zeros(input_dim)

        self._error_history: deque[float] = deque(maxlen=surprise_window)
        self._updated_weights_flag = False

    def _normalize_context(self, context: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
        if context is None:
            context = fallback[: self.hidden_dim] if self.hidden_dim <= len(fallback) else fallback
        context = np.asarray(context, dtype=np.float64).flatten()
        if context.shape[0] != self.hidden_dim:
            if context.shape[0] > self.hidden_dim:
                context = context[: self.hidden_dim]
            else:
                padded = np.zeros(self.hidden_dim)
                padded[: context.shape[0]] = context
                context = padded
        return context

    def _forward(
        self, x: np.ndarray, context: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns combined_pred, h1, h2, pred1, pred2."""
        z1 = self.W1 @ x + self.b1
        h1 = _tanh(z1)
        pred1 = self.W1o @ h1 + self.b1o

        z2 = self.W2 @ context + self.b2
        h2 = _tanh(z2)
        pred2 = self.W2o @ h2 + self.b2o

        combined = 0.5 * pred1 + 0.5 * pred2
        return combined, h1, h2, pred1, pred2

    def _backward(
        self,
        error_vec: np.ndarray,
        x: np.ndarray,
        context: np.ndarray,
        h1: np.ndarray,
        h2: np.ndarray,
        lr: float,
        update_hidden: bool = True,
    ) -> None:
        """SGD — fast output layers; optional slower hidden updates."""
        self.W1o += lr * np.outer(error_vec, h1)
        self.b1o += lr * error_vec
        self.W2o += lr * np.outer(error_vec, h2)
        self.b2o += lr * error_vec

        if not update_hidden:
            return

        hlr = lr * 0.1
        dh1 = (self.W1o.T @ error_vec) * _tanh_derivative(h1)
        self.W1 += hlr * np.outer(dh1, x)
        self.b1 += hlr * dh1

        dh2 = (self.W2o.T @ error_vec) * _tanh_derivative(h2)
        self.W2 += hlr * np.outer(dh2, context)
        self.b2 += hlr * dh2

    def predict(self, x: np.ndarray, context: np.ndarray | None = None) -> dict:
        x = np.asarray(x, dtype=np.float64).flatten()
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[0]}")

        context = self._normalize_context(context, x)
        combined, h1, h2, _, _ = self._forward(x, context)
        error_vec = x - combined
        prediction_error = float(np.sum(error_vec ** 2))

        self._error_history.append(prediction_error)
        self._backward(error_vec, x, context, h1, h2, self.learning_rate)
        self._updated_weights_flag = True

        return {
            "prediction": combined,
            "prediction_error": prediction_error,
            "updated_weights_flag": self._updated_weights_flag,
        }

    def predict_no_learn(self, x: np.ndarray, context: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Forward pass only."""
        x = np.asarray(x, dtype=np.float64).flatten()
        context = self._normalize_context(context, x)
        combined, _, _, _, _ = self._forward(x, context)
        error = float(np.sum((x - combined) ** 2))
        return combined, error

    def predict_with_perturbation(
        self, x: np.ndarray, perturbation: np.ndarray, context: np.ndarray | None = None
    ) -> float:
        x = np.asarray(x, dtype=np.float64).flatten()
        perturbation = np.asarray(perturbation, dtype=np.float64).flatten()
        perturbed = x + perturbation
        _, error = self.predict_no_learn(perturbed, context)
        return error

    def consolidate_belief(
        self,
        target: np.ndarray,
        context: np.ndarray | None = None,
        n_steps: int = 8,
        learning_rate: float | None = None,
    ) -> float:
        """
        Offline mini-consolidation: teach the model that target predicts itself.

        Analogous to hippocampal replay strengthening cortical weights after deliberation.
        Returns final prediction error on target.
        """
        target = np.asarray(target, dtype=np.float64).flatten()
        context = self._normalize_context(context, target)
        lr = learning_rate if learning_rate is not None else self.learning_rate * 5.0

        final_error = 0.0
        for _ in range(n_steps):
            combined, h1, h2, _, _ = self._forward(target, context)
            error_vec = target - combined
            error_vec = np.clip(error_vec, -5.0, 5.0)
            final_error = float(np.sum((target - combined) ** 2))
            self._backward(error_vec, target, context, h1, h2, lr, update_hidden=False)

        return final_error

    def preview_consolidation(
        self,
        target: np.ndarray,
        probe_input: np.ndarray,
        context: np.ndarray | None = None,
        n_steps: int = 8,
        learning_rate: float | None = None,
    ) -> float:
        """
        Simulate consolidation on a weight snapshot, return probe error after replay.

        Does not modify live weights — used by ReasoningLoop to compare hypotheses.
        """
        saved = self.get_weights_dict()
        lr = learning_rate if learning_rate is not None else self.learning_rate * 2.0
        self.consolidate_belief(target, context=context, n_steps=n_steps, learning_rate=lr)
        _, probe_error = self.predict_no_learn(probe_input, context)
        self.load_weights_dict(saved)
        return probe_error

    def get_surprise(self) -> float:
        if not self._error_history:
            return 0.0
        return float(np.mean(self._error_history))

    def get_weights_dict(self) -> dict:
        return {
            "version": "1.3",
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W1o": self.W1o.tolist(),
            "b1o": self.b1o.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "W2o": self.W2o.tolist(),
            "b2o": self.b2o.tolist(),
        }

    def load_weights_dict(self, weights: dict) -> None:
        version = weights.get("version", "1.0")
        if version == "1.2" or "W1o" in weights:
            self.W1 = np.array(weights["W1"], dtype=np.float64)
            self.b1 = np.array(weights["b1"], dtype=np.float64)
            self.W1o = np.array(weights["W1o"], dtype=np.float64)
            self.b1o = np.array(weights["b1o"], dtype=np.float64)
            self.W2 = np.array(weights["W2"], dtype=np.float64)
            self.b2 = np.array(weights["b2"], dtype=np.float64)
            self.W2o = np.array(weights["W2o"], dtype=np.float64)
            self.b2o = np.array(weights["b2o"], dtype=np.float64)
        else:
            # Migrate v1.0/v1.1 linear weights into MLP output layer approximation
            w1_old = np.array(weights["W1"], dtype=np.float64)
            b1_old = np.array(weights["b1"], dtype=np.float64)
            w2_old = np.array(weights["W2"], dtype=np.float64)
            b2_old = np.array(weights["b2"], dtype=np.float64)
            self.W1o = w1_old
            self.b1o = b1_old
            self.W2o = w2_old
            self.b2o = b2_old

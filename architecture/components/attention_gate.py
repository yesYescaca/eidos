"""Attention gating — top-down goal relevance + bottom-up surprise."""

from __future__ import annotations

from typing import Any

import numpy as np


class AttentionGate:
    """Routes signals by salience = alpha * surprise + beta * goal_relevance."""

    def __init__(self, alpha: float = 0.6, beta: float = 0.4) -> None:
        self.alpha = alpha
        self.beta = beta
        self._goal: np.ndarray | None = None
        self._last_salience: list[dict[str, Any]] = []

    def set_goal(self, goal_vector: np.ndarray) -> None:
        self._goal = np.asarray(goal_vector, dtype=np.float64).flatten()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - self._cosine_similarity(a, b)

    def score_signals(
        self,
        signals: list[dict[str, Any]],
        workspace_mean: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Score and rank incoming signals by salience."""
        if workspace_mean is None:
            workspace_mean = np.zeros(64)

        workspace_mean = np.asarray(workspace_mean, dtype=np.float64)
        goal = self._goal if self._goal is not None else workspace_mean

        scored = []
        for signal in signals:
            vec = np.asarray(signal["content_vector"], dtype=np.float64)
            surprise_score = self._cosine_distance(vec, workspace_mean)
            goal_relevance = max(0.0, self._cosine_similarity(vec, goal))
            salience = self.alpha * surprise_score + self.beta * goal_relevance
            scored.append({**signal, "salience": salience, "surprise_score": surprise_score, "goal_relevance": goal_relevance})

        scored.sort(key=lambda s: s["salience"], reverse=True)
        self._last_salience = scored
        return scored

    def get_salience_distribution(self) -> list[float]:
        return [s["salience"] for s in self._last_salience]

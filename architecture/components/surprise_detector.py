"""Relative surprise detection — triggers deliberation on error spikes."""

from __future__ import annotations

from collections import deque

import numpy as np


class SurpriseDetector:
    """
    Detects relative surprise: current error significantly above recent baseline.

    Uses spike ratio (error > mean * ratio) and std threshold (error > mean + k*std).
    Requires minimum history before triggering to avoid cold-start false positives.
    """

    def __init__(
        self,
        window: int = 20,
        spike_ratio: float = 2.0,
        std_multiplier: float = 2.0,
        min_history: int = 5,
    ) -> None:
        self.window = window
        self.spike_ratio = spike_ratio
        self.std_multiplier = std_multiplier
        self.min_history = min_history
        self._history: deque[float] = deque(maxlen=window)

    def update(self, error: float) -> None:
        self._history.append(error)

    def baseline(self) -> float:
        if not self._history:
            return 0.0
        return float(np.mean(self._history))

    def std(self) -> float:
        if len(self._history) < 2:
            return 0.0
        return float(np.std(self._history))

    def surprise_ratio(self, current_error: float) -> float:
        """How many times above baseline the current error is."""
        base = self.baseline()
        if base < 1e-6:
            return current_error
        return current_error / base

    def is_surprising(self, current_error: float) -> bool:
        if len(self._history) < self.min_history:
            return False
        mean = self.baseline()
        std = self.std()
        if mean < 1e-6:
            return current_error > 1.0
        spike = current_error > mean * self.spike_ratio
        std_threshold = current_error > mean + self.std_multiplier * std
        return spike or std_threshold

    def to_dict(self) -> dict:
        return {
            "history": list(self._history),
            "window": self.window,
            "spike_ratio": self.spike_ratio,
            "std_multiplier": self.std_multiplier,
            "min_history": self.min_history,
        }

    def load_dict(self, data: dict) -> None:
        self._history = deque(data.get("history", []), maxlen=self.window)

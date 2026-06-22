"""Fast episodic trace buffer — hippocampal waking experience log."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class EpisodicBuffer:
    """Rolling store of (label, vector) traces for offline sleep replay."""

    def __init__(self, capacity: int = 500) -> None:
        self.capacity = capacity
        self._traces: deque[dict[str, Any]] = deque(maxlen=capacity)

    def record(self, label: str, vector: np.ndarray) -> None:
        self._traces.append({
            "label": label,
            "vector": np.asarray(vector, dtype=np.float64).copy(),
        })

    def sample(self, n: int, rng: np.random.Generator | None = None) -> list[dict[str, Any]]:
        if not self._traces:
            return []
        rng = rng or np.random.default_rng()
        n = min(n, len(self._traces))
        indices = rng.choice(len(self._traces), size=n, replace=True)
        return [dict(self._traces[i]) for i in indices]

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trace in self._traces:
            lbl = trace["label"]
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._traces)

    def clear(self) -> None:
        self._traces.clear()

    def to_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "traces": [
                {"label": t["label"], "vector": t["vector"].tolist()}
                for t in self._traces
            ],
        }

    def load_dict(self, data: dict) -> None:
        self.capacity = data.get("capacity", self.capacity)
        self._traces = deque(maxlen=self.capacity)
        for entry in data.get("traces", []):
            self._traces.append({
                "label": entry["label"],
                "vector": np.array(entry["vector"], dtype=np.float64),
            })

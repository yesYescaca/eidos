"""Recent context tracker — infers recovery target from episodic history."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

import numpy as np


class RecoveryContextTracker:
    """
    Maintains a rolling window of recent (label, vector) steps.

    Infers which registered concept the agent should recover toward after surprise,
    using dominant recent label frequency and mean vector (no external hints).
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self._history: deque[dict[str, Any]] = deque(maxlen=window)

    def record(self, label: str, vector: np.ndarray) -> None:
        self._history.append({
            "label": label,
            "vector": np.asarray(vector, dtype=np.float64).copy(),
        })

    def infer_recovery_probe(
        self,
        concept_vectors: dict[str, np.ndarray],
        current_label: str,
        input_dim: int,
        workspace_labels: list[str] | None = None,
    ) -> tuple[np.ndarray | None, str | None, str]:
        """
        Infer recovery probe vector and source of inference.

        Returns (probe_vector, dominant_label, inference_source).
        """
        if not concept_vectors:
            return None, None, "none"

        counts: Counter[str] = Counter()
        vectors_by_label: dict[str, list[np.ndarray]] = {}

        for entry in self._history:
            label = entry["label"]
            if label not in concept_vectors:
                continue
            counts[label] += 1
            vectors_by_label.setdefault(label, []).append(entry["vector"])

        if counts:
            dominant = counts.most_common(1)[0][0]
            probe = np.mean(vectors_by_label[dominant], axis=0)
            return self._fit_dim(probe, input_dim), dominant, "recent_history"

        if workspace_labels:
            ws_counts: Counter[str] = Counter()
            for label in workspace_labels:
                if label in concept_vectors and label != current_label:
                    ws_counts[label] += 1
            if ws_counts:
                dominant = ws_counts.most_common(1)[0][0]
                return concept_vectors[dominant].copy(), dominant, "workspace"

        if current_label in concept_vectors:
            return concept_vectors[current_label].copy(), current_label, "current_label"

        return None, None, "none"

    def _fit_dim(self, vec: np.ndarray, dim: int) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64).flatten()
        if vec.shape[0] < dim:
            padded = np.zeros(dim)
            padded[: vec.shape[0]] = vec
            return padded
        return vec[:dim]

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "history": [
                {"label": e["label"], "vector": e["vector"].tolist()}
                for e in self._history
            ],
        }

    def load_dict(self, data: dict) -> None:
        self.window = data.get("window", self.window)
        self._history = deque(maxlen=self.window)
        for entry in data.get("history", []):
            self._history.append({
                "label": entry["label"],
                "vector": np.array(entry["vector"], dtype=np.float64),
            })

    def label_count(self, label: str) -> int:
        return sum(1 for e in self._history if e["label"] == label)

    def clear(self) -> None:
        self._history.clear()

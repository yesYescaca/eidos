"""Slow semantic memory — consolidated beliefs (neocortical store)."""

from __future__ import annotations

from typing import Any

import numpy as np


class BeliefGraph:
    """
    Slow store from Complementary Learning Systems theory.

    Accumulates concept strength and prototype vectors via offline sleep replay.
    Used for recovery targeting when recent episodic context is missing or misleading.
    """

    def __init__(self) -> None:
        self._strength: dict[str, float] = {}
        self._prototypes: dict[str, np.ndarray] = {}

    def integrate(self, label: str, vector: np.ndarray, weight: float = 1.0) -> None:
        """Add evidence for a concept (running-mean prototype + strength)."""
        vec = np.asarray(vector, dtype=np.float64).flatten()
        weight = float(weight)
        if label not in self._strength:
            self._strength[label] = weight
            self._prototypes[label] = vec.copy()
            return

        old_strength = self._strength[label]
        new_strength = old_strength + weight
        proto = self._prototypes[label]
        min_len = min(len(proto), len(vec))
        blended = proto.copy()
        blended[:min_len] = (
            proto[:min_len] * old_strength + vec[:min_len] * weight
        ) / new_strength
        self._strength[label] = new_strength
        self._prototypes[label] = blended

    def strength(self, label: str) -> float:
        return self._strength.get(label, 0.0)

    def get_prototype(self, label: str) -> np.ndarray | None:
        return self._prototypes.get(label)

    def dominant_concept(
        self,
        registered: set[str] | None = None,
    ) -> tuple[str | None, float]:
        candidates = self._strength
        if registered is not None:
            candidates = {k: v for k, v in candidates.items() if k in registered}
        if not candidates:
            return None, 0.0
        label = max(candidates, key=candidates.get)
        return label, candidates[label]

    def top_concepts(self, k: int = 3, registered: set[str] | None = None) -> list[str]:
        candidates = self._strength
        if registered is not None:
            candidates = {lbl: s for lbl, s in candidates.items() if lbl in registered}
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [lbl for lbl, _ in ranked[:k]]

    def infer_recovery_probe(
        self,
        concept_vectors: dict[str, np.ndarray],
        input_dim: int,
    ) -> tuple[np.ndarray | None, str | None]:
        registered = set(concept_vectors.keys())
        label, strength = self.dominant_concept(registered=registered)
        if label is None or strength <= 0:
            return None, None

        proto = self._prototypes.get(label)
        if proto is not None:
            return self._fit_dim(proto, input_dim), label
        return concept_vectors[label].copy(), label

    def _fit_dim(self, vec: np.ndarray, dim: int) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64).flatten()
        if vec.shape[0] < dim:
            padded = np.zeros(dim)
            padded[: vec.shape[0]] = vec
            return padded
        return vec[:dim]

    def clear(self) -> None:
        self._strength.clear()
        self._prototypes.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "strength": dict(self._strength),
            "prototypes": {k: v.tolist() for k, v in self._prototypes.items()},
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._strength = {k: float(v) for k, v in data.get("strength", {}).items()}
        self._prototypes = {
            k: np.array(v, dtype=np.float64)
            for k, v in data.get("prototypes", {}).items()
        }

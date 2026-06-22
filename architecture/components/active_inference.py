"""Active inference controller — expected free energy action selection (Friston 2017)."""

from __future__ import annotations

from typing import Any

import numpy as np


class ActiveInferenceController:
    """
    Selects actions that minimise expected free energy (EFE).

    Simplified discrete action space for the EIDOS vector agent:
    - observe: accept current sensory input
    - probe: actively sample a registered concept
    - sleep: offline consolidation when epistemic probes are weak
    """

    def __init__(
        self,
        epistemic_weight: float = 1.0,
        pragmatic_weight: float = 0.6,
        probe_noise_std: float = 0.03,
        sleep_epistemic_bonus: float = 0.15,
        belief_strength_threshold: float = 0.05,
    ) -> None:
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight
        self.probe_noise_std = probe_noise_std
        self.sleep_epistemic_bonus = sleep_epistemic_bonus
        self.belief_strength_threshold = belief_strength_threshold
        self._last_selection: dict[str, Any] | None = None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _epistemic_probe(
        self,
        concept: str,
        concept_vector: np.ndarray,
        observation: np.ndarray,
        context: np.ndarray,
        prediction_engine: Any,
    ) -> float:
        """Expected information gain from probing a concept (closeness-weighted)."""
        _, current_error = prediction_engine.predict_no_learn(observation, context)
        probe_vec = np.asarray(concept_vector, dtype=np.float64).flatten()
        _, probe_error = prediction_engine.predict_no_learn(probe_vec, context)
        closeness = max(0.0, self._cosine_similarity(observation, probe_vec))
        raw_gain = max(0.0, current_error - probe_error)
        return raw_gain * closeness

    def _pragmatic_probe(
        self,
        concept_vector: np.ndarray,
        goal: np.ndarray | None,
    ) -> float:
        if goal is None:
            return 0.0
        return max(0.0, self._cosine_similarity(concept_vector, goal))

    def _expected_free_energy(
        self, epistemic: float, pragmatic: float
    ) -> float:
        """Lower is better — minimise G."""
        value = (
            self.epistemic_weight * epistemic
            + self.pragmatic_weight * pragmatic
        )
        return -value

    def build_candidates(
        self,
        concept_vectors: dict[str, np.ndarray],
        belief_strengths: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = [
            {"type": "observe", "concept": None, "label": "observe"},
        ]
        for concept in sorted(concept_vectors.keys()):
            candidates.append({
                "type": "probe",
                "concept": concept,
                "label": f"probe:{concept}",
            })
        strengths = belief_strengths or {}
        weak_beliefs = not strengths or max(strengths.values(), default=0.0) < self.belief_strength_threshold
        if weak_beliefs and concept_vectors:
            candidates.append({"type": "sleep", "concept": None, "label": "sleep"})
        return candidates

    def select_action(
        self,
        observation: np.ndarray,
        context: np.ndarray,
        concept_vectors: dict[str, np.ndarray],
        prediction_engine: Any,
        goal: np.ndarray | None = None,
        belief_strengths: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Return selected action and scored candidates."""
        candidates = self.build_candidates(concept_vectors, belief_strengths)
        scored: list[dict[str, Any]] = []
        if goal is not None:
            epi_w = self.epistemic_weight * 0.2
            prag_w = self.pragmatic_weight * 4.0
        else:
            epi_w = self.epistemic_weight
            prag_w = self.pragmatic_weight

        for action in candidates:
            if action["type"] == "observe":
                epistemic = 0.0
                pragmatic = 0.0
            elif action["type"] == "probe":
                concept = action["concept"]
                vec = concept_vectors[concept]
                epistemic = self._epistemic_probe(
                    concept, vec, observation, context, prediction_engine
                )
                pragmatic = self._pragmatic_probe(vec, goal)
            else:  # sleep
                epistemic = self.sleep_epistemic_bonus
                pragmatic = 0.0

            efe = -(epi_w * epistemic + prag_w * pragmatic)
            scored.append({
                **action,
                "epistemic_value": epistemic,
                "pragmatic_value": pragmatic,
                "expected_free_energy": efe,
            })

        scored.sort(key=lambda x: (x["expected_free_energy"], -x["pragmatic_value"]))
        if goal is not None:
            probes = [item for item in scored if item["type"] == "probe"]
            selected = max(probes, key=lambda x: x["pragmatic_value"]) if probes else scored[0]
        else:
            selected = scored[0]
        result = {
            "action": selected,
            "candidates": scored,
        }
        self._last_selection = result
        return result

    def get_last_selection(self) -> dict[str, Any] | None:
        return self._last_selection

    def probe_vector(
        self, concept_vector: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample vector for an epistemic probe action."""
        vec = np.asarray(concept_vector, dtype=np.float64).flatten().copy()
        if rng is not None and self.probe_noise_std > 0:
            vec = vec + rng.normal(0, self.probe_noise_std, vec.shape)
        return vec

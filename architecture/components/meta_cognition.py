"""Meta-cognition — monitor context conflicts and reasoning quality (v3.0)."""

from __future__ import annotations

from typing import Any

import numpy as np

from architecture.components.belief_graph import BeliefGraph
from architecture.components.episodic_buffer import EpisodicBuffer


class MetaCognitionMonitor:
    """
    System 2 self-monitoring:
    - (A) Detect misleading short-term context vs long-term episodic evidence
    - (B) Flag ambiguous or low-confidence reasoning outcomes
    """

    def __init__(
        self,
        misleading_context_ratio: float = 2.0,
        low_confidence_threshold: float = 0.4,
        close_hypothesis_epsilon: float = 0.5,
    ) -> None:
        self.misleading_context_ratio = misleading_context_ratio
        self.low_confidence_threshold = low_confidence_threshold
        self.close_hypothesis_epsilon = close_hypothesis_epsilon
        self._last_flags: list[str] = []

    @property
    def last_flags(self) -> list[str]:
        return list(self._last_flags)

    def clear_flags(self) -> None:
        self._last_flags.clear()

    def _fit_dim(self, vec: np.ndarray, dim: int) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64).flatten()
        if vec.shape[0] < dim:
            padded = np.zeros(dim)
            padded[: vec.shape[0]] = vec
            return padded
        return vec[:dim]

    def _probe_for_label(
        self,
        label: str,
        episodic_buffer: EpisodicBuffer,
        concept_vectors: dict[str, np.ndarray],
        input_dim: int,
    ) -> np.ndarray | None:
        mean_vec = episodic_buffer.mean_vector_for_label(label)
        if mean_vec is not None:
            return self._fit_dim(mean_vec, input_dim)
        if label in concept_vectors:
            return concept_vectors[label].copy()
        return None

    def detect_misleading_context(
        self,
        recent_label: str | None,
        recent_window_count: int,
        episodic_buffer: EpisodicBuffer,
        registered: set[str],
    ) -> tuple[bool, str | None, int, int]:
        """
        Compare short recent label vs long episodic distribution.

        Returns (is_misleading, episodic_dominant, dominant_count, recent_label_episodic_count).
        """
        counts = episodic_buffer.label_counts()
        reg_counts = {k: v for k, v in counts.items() if k in registered}
        if not reg_counts or not recent_label or recent_label not in registered:
            return False, None, 0, 0

        episodic_dominant = max(reg_counts, key=reg_counts.get)
        dominant_count = reg_counts[episodic_dominant]
        recent_episodic_count = reg_counts.get(recent_label, 0)

        if recent_label == episodic_dominant:
            return False, episodic_dominant, dominant_count, recent_episodic_count

        long_term_support = dominant_count
        short_term_support = max(recent_window_count, recent_episodic_count)
        is_misleading = long_term_support >= (
            short_term_support * self.misleading_context_ratio
        )
        return is_misleading, episodic_dominant, dominant_count, recent_episodic_count

    def arbitrate_recovery(
        self,
        recent_probe: np.ndarray | None,
        recent_label: str | None,
        recent_source: str,
        recent_window_count: int,
        belief_probe: np.ndarray | None,
        belief_label: str | None,
        belief_strength_fn,
        episodic_buffer: EpisodicBuffer,
        concept_vectors: dict[str, np.ndarray],
        input_dim: int,
        workspace_labels: list[str],
        goal_probe: tuple[np.ndarray, str] | None,
        slow_store_conflict_ratio: float,
    ) -> tuple[np.ndarray | None, str | None, str, list[str]]:
        """
        Resolve recovery probe with meta-cognitive override for misleading context.

        Returns (probe, label, source, meta_flags).
        """
        flags: list[str] = []
        registered = set(concept_vectors.keys())

        if recent_label and recent_label in registered:
            is_misleading, episodic_dominant, _, _ = self.detect_misleading_context(
                recent_label,
                recent_window_count,
                episodic_buffer,
                registered,
            )
            if is_misleading and episodic_dominant:
                flags.append("misleading_context_detected")
                probe = self._probe_for_label(
                    episodic_dominant, episodic_buffer, concept_vectors, input_dim
                )
                if probe is not None:
                    self._last_flags = flags
                    return probe, episodic_dominant, "meta_cognition", flags

        if belief_probe is not None and recent_probe is not None:
            if belief_label != recent_label:
                belief_strength = belief_strength_fn(belief_label or "")
                recent_strength = belief_strength_fn(recent_label or "")
                recent_count = float(recent_window_count)
                slow_wins = belief_strength > max(
                    recent_strength, recent_count
                ) * slow_store_conflict_ratio
                if slow_wins:
                    if belief_label != recent_label:
                        flags.append("belief_over_recent_conflict")
                    self._last_flags = flags
                    return belief_probe, belief_label, "belief_graph", flags
            self._last_flags = flags
            return recent_probe, recent_label, recent_source, flags

        if belief_probe is not None:
            self._last_flags = flags
            return belief_probe, belief_label, "belief_graph", flags

        if recent_probe is not None:
            self._last_flags = flags
            return recent_probe, recent_label, recent_source, flags

        episodic_dominant, dominant_count = episodic_buffer.dominant_label(registered)
        if episodic_dominant and dominant_count > 0:
            probe = self._probe_for_label(
                episodic_dominant, episodic_buffer, concept_vectors, input_dim
            )
            if probe is not None:
                flags.append("episodic_fallback")
                self._last_flags = flags
                return probe, episodic_dominant, "episodic_memory", flags

        if goal_probe is not None:
            probe, label = goal_probe
            self._last_flags = flags
            return probe, label, "goal", flags

        self._last_flags = flags
        return None, None, "none", flags

    def evaluate_reasoning(
        self,
        hypothesis: dict[str, Any] | None,
        trace_entry: dict[str, Any] | None,
    ) -> list[str]:
        """(B) Inspect reasoning outcome for ambiguity and low confidence."""
        flags: list[str] = []
        if not hypothesis:
            self._last_flags = list(set(self._last_flags))
            return flags

        confidence = float(hypothesis.get("confidence", 0.0))
        if confidence < self.low_confidence_threshold:
            flags.append("low_confidence")

        if trace_entry and "hypotheses" in trace_entry:
            ranked = sorted(
                trace_entry["hypotheses"],
                key=lambda h: h.get("predicted_error", float("inf")),
            )
            if len(ranked) >= 2:
                best = ranked[0].get("predicted_error", 0.0)
                runner_up = ranked[1].get("predicted_error", 0.0)
                if abs(runner_up - best) < self.close_hypothesis_epsilon:
                    flags.append("ambiguous_hypothesis")

        self._last_flags = list(dict.fromkeys(self._last_flags + flags))
        return flags

    def should_suppress_hypothesis(self, flags: list[str]) -> bool:
        """Defer hypothesis application when reasoning is unreliable."""
        return "ambiguous_hypothesis" in flags and "low_confidence" in flags

"""Offline sleep replay — hippocampal → neocortical consolidation."""

from __future__ import annotations

from typing import Any

import numpy as np

from architecture.components.association_store import AssociationStore
from architecture.components.belief_graph import BeliefGraph
from architecture.components.episodic_buffer import EpisodicBuffer
from architecture.components.prediction_engine import PredictionEngine


class SleepReplay:
    """
    Sample episodic traces and integrate into the slow BeliefGraph.

    Optionally reinforces associations and consolidates prediction weights
    for dominant concepts (hippocampal replay strengthening cortex).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._last_trace: list[dict[str, Any]] = []

    def run(
        self,
        episodic_buffer: EpisodicBuffer,
        belief_graph: BeliefGraph,
        associations: AssociationStore,
        prediction_engine: PredictionEngine | None = None,
        concept_vectors: dict[str, np.ndarray] | None = None,
        n_replays: int = 50,
        integrate_weight: float = 1.0,
        association_boost: float = 0.05,
        consolidate_top_k: int = 2,
        consolidate_steps: int = 10,
        consolidate_lr: float = 0.02,
    ) -> dict[str, Any]:
        traces = episodic_buffer.sample(n_replays, rng=self._rng)
        if not traces:
            return {"replays": 0, "concepts_strengthened": [], "associations_updated": 0}

        labels_replayed: list[str] = []
        for trace in traces:
            label = trace["label"]
            belief_graph.integrate(label, trace["vector"], weight=integrate_weight)
            labels_replayed.append(label)

        assoc_updates = 0
        unique_window: list[str] = []
        for label in labels_replayed:
            unique_window.append(label)
            if len(unique_window) >= 3:
                batch = list(dict.fromkeys(unique_window[-3:]))
                if len(batch) >= 2:
                    associations.hebbian_update(batch)
                    graph = associations.to_dict()
                    for i, a in enumerate(batch):
                        for b in batch[i + 1 :]:
                            if a in graph and b in graph.get(a, {}):
                                graph[a][b] += association_boost
                                graph[b][a] = graph[a][b]
                    associations.load_dict(graph)
                    assoc_updates += 1

        concepts_consolidated: list[str] = []
        if prediction_engine is not None and concept_vectors:
            registered = set(concept_vectors.keys())
            for label in belief_graph.top_concepts(k=consolidate_top_k, registered=registered):
                vec = concept_vectors[label]
                ctx = vec[: prediction_engine.hidden_dim]
                prediction_engine.consolidate_belief(
                    target=vec,
                    context=ctx,
                    n_steps=consolidate_steps,
                    learning_rate=consolidate_lr,
                )
                concepts_consolidated.append(label)

        self._last_trace = [
            {"label": t["label"], "replayed": True} for t in traces[:10]
        ]
        return {
            "replays": len(traces),
            "concepts_strengthened": list(dict.fromkeys(labels_replayed)),
            "dominant_after_sleep": belief_graph.dominant_concept(registered=set(concept_vectors or {}))[0],
            "associations_updated": assoc_updates,
            "prediction_consolidated": concepts_consolidated,
        }

    def get_last_trace(self) -> list[dict[str, Any]]:
        return list(self._last_trace)

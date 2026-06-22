"""Fast episodic memory — Hebbian association graph."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any


class AssociationStore:
    """Weighted adjacency graph with Hebbian learning and decay."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        decay_factor: float = 0.99,
        prune_threshold: float = 0.01,
    ) -> None:
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.prune_threshold = prune_threshold
        self._graph: dict[str, dict[str, float]] = defaultdict(dict)

    def hebbian_update(self, active_concepts: list[str]) -> None:
        """Strengthen associations between co-active concepts."""
        unique = list(dict.fromkeys(active_concepts))
        for i, a in enumerate(unique):
            for b in unique[i + 1 :]:
                current = self._graph[a].get(b, 0.0)
                self._graph[a][b] = current + self.learning_rate
                self._graph[b][a] = self._graph[a][b]

    def apply_decay(self) -> None:
        """Multiply all weights by decay factor and prune weak edges."""
        to_remove: list[tuple[str, str]] = []
        for a, neighbors in list(self._graph.items()):
            for b, weight in list(neighbors.items()):
                new_weight = weight * self.decay_factor
                if new_weight < self.prune_threshold:
                    to_remove.append((a, b))
                else:
                    self._graph[a][b] = new_weight
        for a, b in to_remove:
            if b in self._graph.get(a, {}):
                del self._graph[a][b]
            if a in self._graph.get(b, {}):
                del self._graph[b][a]
            if a in self._graph and not self._graph[a]:
                del self._graph[a]
            if b in self._graph and not self._graph[b]:
                del self._graph[b]

    def get_associates(self, concept: str, top_k: int = 5) -> list[tuple[str, float]]:
        neighbors = self._graph.get(concept, {})
        sorted_pairs = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:top_k]

    def get_strongest_path(self, a: str, b: str) -> list[str] | None:
        """BFS path maximising minimum edge weight along the path."""
        if a == b:
            return [a]
        if a not in self._graph:
            return None

        queue: deque[tuple[str, list[str], float]] = deque([(a, [a], float("inf"))])
        visited = {a}
        best_path: list[str] | None = None
        best_min_weight = -1.0

        while queue:
            node, path, min_w = queue.popleft()
            for neighbor, weight in self._graph.get(node, {}).items():
                if neighbor in visited:
                    continue
                new_min = min(min_w, weight)
                new_path = path + [neighbor]
                if neighbor == b:
                    if new_min > best_min_weight:
                        best_min_weight = new_min
                        best_path = new_path
                else:
                    visited.add(neighbor)
                    queue.append((neighbor, new_path, new_min))

        return best_path

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {a: dict(neighbors) for a, neighbors in self._graph.items()}

    def load_dict(self, data: dict[str, dict[str, float]]) -> None:
        self._graph = defaultdict(dict)
        for a, neighbors in data.items():
            for b, w in neighbors.items():
                self._graph[a][b] = float(w)

    def get_all_edges(self) -> list[dict[str, Any]]:
        edges = []
        seen = set()
        for a, neighbors in self._graph.items():
            for b, w in neighbors.items():
                key = tuple(sorted([a, b]))
                if key not in seen:
                    seen.add(key)
                    edges.append({"source": a, "target": b, "weight": w})
        return edges

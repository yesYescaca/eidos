"""Limited-capacity active memory — Global Workspace implementation."""

from __future__ import annotations

import uuid
from typing import Any


class WorkspaceBuffer:
    """Fixed-capacity workspace where active representations are globally broadcast."""

    def __init__(self, capacity: int = 7) -> None:
        self.capacity = capacity
        self._slots: list[dict[str, Any]] = []

    def _age_all(self) -> None:
        for slot in self._slots:
            slot["age"] += 1

    def _eviction_score(self, slot: dict[str, Any]) -> float:
        age = max(slot["age"], 1)
        return slot["salience"] * (1.0 / age)

    def insert(self, content: Any, salience: float, source: str) -> str:
        """Insert content into workspace. Evicts lowest score if at capacity."""
        self._age_all()
        slot_id = str(uuid.uuid4())[:8]
        new_slot = {
            "id": slot_id,
            "content": content,
            "salience": float(salience),
            "age": 1,
            "source": source,
        }
        if len(self._slots) >= self.capacity:
            evict_idx = min(
                range(len(self._slots)),
                key=lambda i: self._eviction_score(self._slots[i]),
            )
            self._slots.pop(evict_idx)
        self._slots.append(new_slot)
        return slot_id

    def broadcast(self) -> list[dict[str, Any]]:
        """Return all active slots (globally readable)."""
        return [dict(s) for s in self._slots]

    def get_by_id(self, slot_id: str) -> dict[str, Any] | None:
        for slot in self._slots:
            if slot["id"] == slot_id:
                return dict(slot)
        return None

    def evict_by_label(self, label: str) -> int:
        """Remove all slots whose content label matches. Returns count evicted."""
        before = len(self._slots)
        self._slots = [
            s for s in self._slots
            if not (
                (isinstance(s["content"], dict) and s["content"].get("label") == label)
                or s["content"] == label
            )
        ]
        return before - len(self._slots)

    def clear(self) -> None:
        self._slots.clear()

    def __len__(self) -> int:
        return len(self._slots)

    def __repr__(self) -> str:
        if not self._slots:
            return "WorkspaceBuffer [empty]"
        lines = ["WorkspaceBuffer", "┌" + "─" * 58 + "┐"]
        lines.append(f"│ {'ID':<8} {'Salience':>8} {'Age':>4} {'Source':<12} Content")
        lines.append("├" + "─" * 58 + "┤")
        for s in self._slots:
            content_str = str(s["content"])
            if len(content_str) > 18:
                content_str = content_str[:15] + "..."
            lines.append(
                f"│ {s['id']:<8} {s['salience']:>8.3f} {s['age']:>4} "
                f"{s['source']:<12} {content_str}"
            )
        lines.append("└" + "─" * 58 + "┘")
        return "\n".join(lines)

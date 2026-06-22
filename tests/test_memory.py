import numpy as np

from architecture.components.working_memory import WorkspaceBuffer


def test_insert_adds_slot():
    ws = WorkspaceBuffer(capacity=3)
    slot_id = ws.insert("hello", salience=0.8, source="test")
    assert len(ws) == 1
    assert ws.get_by_id(slot_id) is not None


def test_eviction_at_capacity():
    ws = WorkspaceBuffer(capacity=2)
    ws.insert("low", salience=0.1, source="a")
    ws.insert("high", salience=0.9, source="b")
    ws.insert("newer", salience=0.5, source="c")
    contents = [s["content"] for s in ws.broadcast()]
    assert "low" not in contents
    assert len(ws) == 2


def test_broadcast_returns_all_active():
    ws = WorkspaceBuffer(capacity=5)
    ws.insert("a", 0.5, "s1")
    ws.insert("b", 0.6, "s2")
    broadcast = ws.broadcast()
    assert len(broadcast) == 2
    assert all("content" in s for s in broadcast)


def test_evict_by_label():
    ws = WorkspaceBuffer(capacity=5)
    ws.insert({"label": "anomaly", "vector": [1]}, 0.9, "input")
    ws.insert({"label": "fire", "vector": [2]}, 0.8, "input")
    removed = ws.evict_by_label("anomaly")
    assert removed == 1
    labels = [s["content"]["label"] for s in ws.broadcast()]
    assert "anomaly" not in labels
    assert "fire" in labels


def test_clear():
    ws = WorkspaceBuffer()
    ws.insert("x", 0.5, "s")
    ws.clear()
    assert len(ws) == 0

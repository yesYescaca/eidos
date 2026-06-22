from architecture.components.association_store import AssociationStore


def test_hebbian_strengthens_pairs():
    store = AssociationStore(learning_rate=0.5, decay_factor=1.0)
    store.hebbian_update(["cat", "dog"])
    associates = store.get_associates("cat")
    assert len(associates) == 1
    assert associates[0][0] == "dog"
    assert associates[0][1] == 0.5


def test_decay_reduces_weights():
    store = AssociationStore(learning_rate=0.5, decay_factor=0.5, prune_threshold=0.001)
    store.hebbian_update(["a", "b"])
    weight_before = store.get_associates("a")[0][1]
    store.apply_decay()
    weight_after = store.get_associates("a")[0][1]
    assert weight_after < weight_before


def test_get_associates_sorted():
    store = AssociationStore(learning_rate=0.3, decay_factor=1.0)
    store.hebbian_update(["x", "a"])
    store.hebbian_update(["x", "b"])
    store.hebbian_update(["x", "b"])
    associates = store.get_associates("x", top_k=5)
    weights = [w for _, w in associates]
    assert weights == sorted(weights, reverse=True)

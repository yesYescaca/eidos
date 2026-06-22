import numpy as np

from architecture.components.surprise_detector import SurpriseDetector


def test_not_surprising_with_insufficient_history():
    det = SurpriseDetector(min_history=5)
    for e in [10, 12, 11, 10, 13]:
        det.update(e)
    assert not det.is_surprising(11)


def test_surprising_on_spike():
    det = SurpriseDetector(spike_ratio=2.0, min_history=5)
    for e in [10, 10, 10, 10, 10, 10]:
        det.update(e)
    assert det.is_surprising(25)


def test_surprise_ratio():
    det = SurpriseDetector(min_history=3)
    for e in [5, 5, 5]:
        det.update(e)
    assert abs(det.surprise_ratio(15) - 3.0) < 0.01

import numpy as np

from cstar.objectives.metrics import jain_fairness


def test_jain_range():
    r = np.array([1.0, 1.0, 1.0])
    assert 0.99 < jain_fairness(r) <= 1.0

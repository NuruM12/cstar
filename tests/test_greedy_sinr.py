import numpy as np

from cstar.schedulers.greedy_sinr import GreedySINR


def test_greedy_respects_budget_and_caps():
    rng = np.random.default_rng(0)
    n = 24
    capacity = np.clip(rng.normal(5.0, 1.0, size=n), 0.5, None)
    # Huge demand so caps are very large (sum caps >> B)
    demand = capacity * 1e6
    B = 50.0

    alloc = GreedySINR().allocate(demand, capacity, B, ctx={})
    assert np.isfinite(alloc).all()
    assert (alloc >= -1e-9).all()
    assert abs(alloc.sum() - B) < 1e-6
    # Never exceeds per-user caps
    caps = np.divide(demand, capacity, out=np.full_like(alloc, np.inf), where=capacity > 0)
    assert (alloc <= caps + 1e-9).all()


def test_greedy_prioritizes_higher_capacity():
    # Highest capacity gets everything until budget or its cap
    c = np.array([10.0, 5.0, 1.0])
    d = c * 1e6  # large caps
    B = 7.0
    alloc = GreedySINR().allocate(d, c, B, ctx={})
    assert np.allclose(alloc, np.array([7.0, 0.0, 0.0]), atol=1e-9)


def test_greedy_redistributes_when_top_caps_are_tight():
    # Tight caps on best users; leftover goes to next ones.
    c = np.array([10.0, 9.0, 8.0])
    caps = np.array([2.0, 1.0, 1000.0])
    d = caps * c  # choose demand so that demand/capacity=caps
    B = 10.0
    alloc = GreedySINR().allocate(d, c, B, ctx={})
    assert abs(alloc.sum() - B) < 1e-6
    # First two saturate at 2 and 1; remaining 7 to third
    assert np.isclose(alloc[0], 2.0, atol=1e-9)
    assert np.isclose(alloc[1], 1.0, atol=1e-9)
    assert np.isclose(alloc[2], 7.0, atol=1e-9)


def test_greedy_zero_capacity_gets_zero():
    d = np.array([100.0, 100.0, 100.0])
    c = np.array([5.0, 0.0, 5.0])
    B = 10.0
    alloc = GreedySINR().allocate(d, c, B, ctx={})
    assert abs(alloc.sum() - B) < 1e-6 or alloc.sum() <= B + 1e-6
    assert alloc[1] == 0.0

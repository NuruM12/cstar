import numpy as np

from cstar.schedulers.proportional import Proportional


def test_proportional_basic_budget_and_nonnegativity():
    rng = np.random.default_rng(0)
    n = 24
    demand = rng.lognormal(2.0, 0.6, size=n)
    sinr_db = rng.normal(10.0, 5.0, size=n).clip(-5, 30)
    capacity = np.log2(1.0 + 10 ** (sinr_db / 10))
    B = 20.0

    alloc = Proportional().allocate(demand, capacity, B, ctx={})
    assert np.isfinite(alloc).all()
    assert (alloc >= -1e-9).all()
    assert abs(alloc.sum() - B) < 1e-6


def test_proportional_honors_ratios_when_uncapped():
    demand = np.array([1.0, 3.0])  # weights 1:3
    capacity = np.array([10.0, 10.0])
    B = 40.0
    alloc = Proportional().allocate(demand, capacity, B, ctx={})
    assert abs(alloc[0] - 10.0) < 1e-9
    assert abs(alloc[1] - 30.0) < 1e-9
    assert abs(alloc.sum() - B) < 1e-6


def test_proportional_zero_demand_fallback_equal_split_over_positive_capacity():
    demand = np.array([0.0, 0.0, 0.0])
    capacity = np.array([5.0, 0.0, 5.0])  # middle user cannot be served
    B = 10.0
    alloc = Proportional().allocate(demand, capacity, B, ctx={})
    assert abs(alloc[0] - 5.0) < 1e-9
    assert abs(alloc[1] - 0.0) < 1e-12
    assert abs(alloc[2] - 5.0) < 1e-9
    assert abs(alloc.sum() - B) < 1e-6


def test_proportional_redistributes_when_some_caps_are_tight():
    # Equal weights, but explicit tight caps on first two users.
    demand = np.array([1.0, 1.0, 1.0])  # equal weights
    capacity = np.array([1.0, 1.0, 1.0])
    caps = np.array([2.0, 1.0, 1000.0])  # tight caps for users 0 and 1
    B = 6.0
    alloc = Proportional().allocate(demand, capacity, B, ctx={"caps": caps})
    assert abs(alloc.sum() - B) < 1e-6
    # First two saturate, leftover goes to third ⇒ [2,1,3]
    assert np.isclose(alloc[0], 2.0, atol=1e-9)
    assert np.isclose(alloc[1], 1.0, atol=1e-9)
    assert np.isclose(alloc[2], 3.0, atol=1e-9)


def test_proportional_demand_power_emphasis():
    demand = np.array([1.0, 3.0])
    capacity = np.array([10.0, 10.0])
    B = 40.0
    alloc_p1 = Proportional().allocate(demand, capacity, B, ctx={"demand_power": 1.0})
    alloc_p2 = Proportional().allocate(demand, capacity, B, ctx={"demand_power": 2.0})
    assert alloc_p2[1] > alloc_p1[1]
    assert abs(alloc_p1.sum() - B) < 1e-6 and abs(alloc_p2.sum() - B) < 1e-6

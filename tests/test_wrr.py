import numpy as np

from cstar.schedulers.wrr import WRR


def test_wrr_respects_budget_and_caps():
    rng = np.random.default_rng(0)
    n = 16
    capacity = np.clip(rng.normal(5.0, 1.0, size=n), 0.5, None)
    demand = np.full(n, 1e6)
    B = 50.0

    alloc = WRR().allocate(demand, capacity, B, ctx={"lambda": 0.5})
    assert np.isfinite(alloc).all()
    assert (alloc >= -1e-9).all()
    assert abs(alloc.sum() - B) < 1e-6


def test_wrr_honors_weight_ratios_when_not_capped():
    demand = np.array([1e6, 1e6])
    capacity = np.array([10.0, 10.0])
    B = 40.0
    weights = np.array([1.0, 3.0])

    alloc = WRR().allocate(demand, capacity, B, ctx={"weights": weights})
    assert np.isclose(alloc[0], 10.0, atol=1e-9)
    assert np.isclose(alloc[1], 30.0, atol=1e-9)


def test_wrr_redistributes_when_some_caps_are_tight():
    capacity = np.array([1.0, 1.0, 1.0])
    demand = np.array([2.0, 1.0, 1000.0])  # caps=[2,1,1000]
    B = 5.0
    alloc = WRR().allocate(demand, capacity, B, ctx={"weights": np.ones(3)})
    assert abs(alloc.sum() - B) < 1e-6
    assert np.isclose(alloc[0], 2.0, atol=1e-9)
    assert np.isclose(alloc[1], 1.0, atol=1e-9)
    assert np.isclose(alloc[2], 2.0, atol=1e-9)


def test_wrr_zero_capacity_gets_zero_bandwidth():
    demand = np.array([100.0, 100.0, 100.0])
    capacity = np.array([5.0, 0.0, 5.0])
    B = 10.0
    alloc = WRR().allocate(demand, capacity, B, ctx={})
    assert abs(alloc.sum() - B) < 1e-6
    assert alloc[1] == 0.0

import numpy as np

from cstar.schedulers.optimal_fair import OptimalFair


def test_optimal_fair_equalizes_rates_when_not_capped():
    rng = np.random.default_rng(0)
    n = 24
    # Large demand → effectively uncapped
    capacity = np.clip(rng.normal(6.0, 2.0, size=n), 0.5, None)
    demand = capacity * 1e6
    B = 40.0

    alloc = OptimalFair().allocate(demand, capacity, B, ctx={})
    assert np.isfinite(alloc).all()
    assert (alloc >= -1e-9).all()
    assert abs(alloc.sum() - B) < 1e-6

    # Non-capped users should have nearly equal *rates*
    caps = np.divide(demand, capacity, out=np.full_like(alloc, np.inf), where=capacity > 0)
    free = (capacity > 0) & (alloc < caps - 1e-6)
    rates = alloc[free] * capacity[free]
    if rates.size >= 2:
        assert rates.max() - rates.min() < 1e-2


def test_optimal_fair_zero_capacity_gets_zero_bandwidth():
    demand = np.array([100.0, 100.0, 100.0])
    capacity = np.array([5.0, 0.0, 5.0])
    B = 10.0
    alloc = OptimalFair().allocate(demand, capacity, B, ctx={})
    assert abs(alloc.sum() - B) < 1e-6 or alloc.sum() <= B + 1e-9
    assert alloc[1] == 0.0


def test_optimal_fair_total_caps_less_than_budget():
    # Two users with tiny caps; B cannot be fully used.
    capacity = np.array([2.0, 2.0])
    demand = np.array([1.0, 1.0])  # caps = [0.5, 0.5]
    B = 10.0
    alloc = OptimalFair().allocate(demand, capacity, B, ctx={})
    assert np.allclose(alloc, np.array([0.5, 0.5]), atol=1e-9)
    assert abs(alloc.sum() - 1.0) < 1e-9


def test_optimal_fair_redistributes_under_tight_caps():
    # Three users, equal capacities; two have tight caps → leftover to the third.
    capacity = np.array([1.0, 1.0, 1.0])
    caps = np.array([2.0, 1.0, 1000.0])
    demand = caps * capacity
    B = 6.0
    alloc = OptimalFair().allocate(demand, capacity, B, ctx={})
    assert abs(alloc.sum() - B) < 1e-6
    assert np.isclose(alloc[0], 2.0, atol=1e-9)
    assert np.isclose(alloc[1], 1.0, atol=1e-9)
    assert np.isclose(alloc[2], 3.0, atol=1e-9)


def test_optimal_fair_rate_increases_with_budget_for_free_users():
    # As B grows, the common rate for non-capped users should not decrease.
    capacity = np.array([4.0, 3.0, 5.0, 2.0])
    demand = capacity * 1e6  # effectively uncapped
    B1, B2 = 10.0, 20.0
    a1 = OptimalFair().allocate(demand, capacity, B1, ctx={})
    a2 = OptimalFair().allocate(demand, capacity, B2, ctx={})

    # Free users (all of them here) get higher (or equal) rate = b*c
    r1 = a1 * capacity
    r2 = a2 * capacity
    assert r2.min() >= r1.min() - 1e-6

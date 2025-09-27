import numpy as np

from cstar.schedulers.uniform import Uniform


def test_uniform_respects_budget_and_caps():
    rng = np.random.default_rng(0)
    n = 32
    capacity = np.clip(rng.normal(5.0, 1.0, size=n), 0.5, None)
    demand = capacity * 1e6  # effectively uncapped
    B = 80.0

    alloc = Uniform().allocate(demand, capacity, B, ctx={})
    assert np.isfinite(alloc).all()
    assert (alloc >= -1e-9).all()
    assert abs(alloc.sum() - B) < 1e-6
    caps = np.divide(demand, capacity, out=np.full_like(alloc, np.inf), where=capacity > 0)
    assert (alloc <= caps + 1e-9).all()


def test_uniform_equal_share_when_uncapped():
    # All serviceable users should split equally when not capped
    demand = np.array([1e6, 1e6, 1e6, 1e6])
    capacity = np.array([5.0, 4.0, 3.0, 2.0])  # all > 0
    B = 40.0
    alloc = Uniform().allocate(demand, capacity, B, ctx={})
    assert np.allclose(alloc, np.array([10.0, 10.0, 10.0, 10.0]), atol=1e-9)


def test_uniform_zero_capacity_gets_zero_and_others_split():
    demand = np.full(3, 1e6)
    capacity = np.array([5.0, 0.0, 5.0])  # middle is not serviceable
    B = 10.0
    alloc = Uniform().allocate(demand, capacity, B, ctx={})
    assert abs(alloc.sum() - B) < 1e-6
    assert alloc[1] == 0.0
    assert np.allclose([alloc[0], alloc[2]], [5.0, 5.0], atol=1e-9)


def test_uniform_redistributes_when_some_caps_are_tight():
    # Three users, equal-share baseline would be 2 each for B=6,
    # but user 1 cap=2, user 2 cap=1 → leftover goes to user 3.
    capacity = np.array([1.0, 1.0, 1.0])
    caps = np.array([2.0, 1.0, 1000.0])
    demand = caps * capacity  # produce those caps via demand/capacity
    B = 6.0
    alloc = Uniform().allocate(demand, capacity, B, ctx={})
    assert abs(alloc.sum() - B) < 1e-6
    assert np.isclose(alloc[0], 2.0, atol=1e-9)
    assert np.isclose(alloc[1], 1.0, atol=1e-9)
    assert np.isclose(alloc[2], 3.0, atol=1e-9)


def test_uniform_ignores_zero_total_demand_caps():
    # If demand.sum()==0 we treat caps as infinite (except c<=0).
    demand = np.array([0.0, 0.0, 0.0, 0.0])
    capacity = np.array([2.0, 2.0, 0.0, 2.0])  # one non-serviceable
    B = 9.0
    alloc = Uniform().allocate(demand, capacity, B, ctx={})
    # Split equally among 3 serviceable users
    assert np.allclose(alloc, np.array([3.0, 3.0, 0.0, 3.0]), atol=1e-9)

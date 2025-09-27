import numpy as np

from cstar.objectives.metrics import jain_fairness
from cstar.schedulers.cstarpp import CStarPP


def test_cstarpp_respects_budget_caps_nonnegativity():
    rng = np.random.default_rng(3)
    n = 20
    demand = np.clip(rng.lognormal(2.0, 0.7, size=n), 0.2, 50.0)
    sinr_db = np.clip(rng.normal(10.0, 6.0, size=n), -5, 30)
    capacity = np.log2(1.0 + 10 ** (sinr_db / 10.0))
    B = 40.0

    alloc = CStarPP().allocate(demand, capacity, B, ctx={})
    caps = np.divide(demand, capacity, out=np.full_like(demand, np.inf), where=capacity > 0)

    assert np.isfinite(alloc).all()
    assert (alloc >= -1e-9).all()
    assert abs(alloc.sum() - B) < 1e-6
    assert (alloc <= caps + 1e-9).all()


def test_cstarpp_manual_weights_behave_as_expected():
    # Two users, very different capacities; large demand so caps non-binding
    demand = np.array([1e6, 1e6])
    capacity = np.array([1.0, 9.0])
    B = 100.0

    # Heavily WRR → skew toward the strong channel (user 2)
    a_wrr = CStarPP().allocate(demand, capacity, B, ctx={"weights": [0.0, 0.0, 1.0, 0.0]})
    assert a_wrr[1] > a_wrr[0]

    # Heavily OptimalFair → equalize *rates*: b_i * c_i nearly equal
    a_fair = CStarPP().allocate(demand, capacity, B, ctx={"weights": [0.0, 0.0, 0.0, 1.0]})
    rates = a_fair * capacity
    assert abs(rates[0] - rates[1]) < 1e-6

    # Heavily Uniform → equal bandwidths
    a_uni = CStarPP().allocate(demand, capacity, B, ctx={"weights": [1.0, 0.0, 0.0, 0.0]})
    assert abs(a_uni[0] - a_uni[1]) < 1e-9


def test_cstarpp_auto_gate_favors_fairness_under_demand_skew():
    # Strong demand skew (one user huge), capacities similar
    demand = np.array([1.0, 100.0, 100.0, 100.0])
    capacity = np.array([5.0, 5.0, 5.0, 5.0])
    B = 40.0

    # Composite with auto gate vs a throughput-leaning mix (WRR)
    comp = CStarPP().allocate(demand, capacity, B, ctx={})
    # simulate a WRR-like tilt by hand to compare fairness
    wrry = CStarPP().allocate(demand, capacity, B, ctx={"weights": [0.0, 0.2, 0.8, 0.0]})

    f_comp = jain_fairness(comp * capacity)
    f_wrry = jain_fairness(wrry * capacity)
    assert f_comp >= f_wrry - 1e-9


def test_cstarpp_auto_gate_exploits_capacity_spread_when_skew_low():
    # Low demand skew, large capacity spread → expect more opportunistic mix than Uniform
    demand = np.array([10.0, 10.0, 10.0, 10.0])
    capacity = np.array([1.0, 3.0, 6.0, 12.0])
    B = 40.0

    a_comp = CStarPP().allocate(demand, capacity, B, ctx={})
    a_uni = CStarPP().allocate(demand, capacity, B, ctx={"weights": [1.0, 0.0, 0.0, 0.0]})

    # Throughput comparison (sum rate)
    t_comp = float((a_comp * capacity).sum())
    t_uni = float((a_uni * capacity).sum())
    assert t_comp >= t_uni - 1e-9

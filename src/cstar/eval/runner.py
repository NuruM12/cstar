from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..objectives.metrics import (
    cvar_shortfall,
    entropy_alloc,
    jain_fairness,
    spectral_efficiency_sum,
    stability_L2,
    throughput_sum,
    unmet_demand_U,
)
from ..schedulers.cstarpp import CStarPP
from ..schedulers.greedy_sinr import GreedySINR
from ..schedulers.optimal_fair import OptimalFair
from ..schedulers.proportional import Proportional
from ..schedulers.uniform import Uniform
from ..schedulers.wrr import WRR

REGISTRY = {
    "uniform": Uniform(),
    "proportional": Proportional(),
    "greedy_sinr": GreedySINR(),
    "wrr": WRR(),
    "optimal_fair": OptimalFair(),
    "cstarpp": CStarPP(),
}


@dataclass
class Snapshot:
    demand: np.ndarray
    capacity: np.ndarray
    B: float


def sample_inputs(
    n: int = 12, B: float = 20.0, seed: int = 123
) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    demand = np.clip(rng.lognormal(mean=2.0, sigma=0.6, size=n), 0.5, 40.0)
    sinr_db = np.clip(rng.normal(10.0, 5.0, size=n), -5.0, 30.0)
    capacity = np.log2(1.0 + 10 ** (sinr_db / 10.0))
    return demand, capacity, B


def gen_snapshots(
    n_snaps: int = 200, n_users: int = 12, B: float = 20.0, seed: int = 123
) -> List[Snapshot]:
    rng = np.random.default_rng(seed)
    snaps: List[Snapshot] = []
    for _ in range(n_snaps):
        d, c, Bb = sample_inputs(n=n_users, B=B, seed=int(rng.integers(0, 1_000_000)))
        snaps.append(Snapshot(d, c, Bb))
    return snaps


def evaluate_schedulers(
    n_snaps: int = 200,
    n_users: int = 12,
    B: float = 20.0,
    seed: int = 123,
    schedulers: List[str] | None = None,
) -> pd.DataFrame:
    snaps = gen_snapshots(n_snaps, n_users, B, seed)
    rows = []
    for sname in schedulers or list(REGISTRY.keys()):
        sch = REGISTRY[sname]
        b_prev = None
        for snap in snaps:
            b = sch.allocate(snap.demand, snap.capacity, snap.B, ctx={})
            r = b * snap.capacity
            row = dict(
                scheduler=sname,
                users=len(b),
                B=snap.B,
                throughput=throughput_sum(r),
                fairness=jain_fairness(r),
                unmet=unmet_demand_U(r, snap.demand),
                cvar=cvar_shortfall(r, snap.demand, alpha=0.8),
                entropy=entropy_alloc(b),
                stability=stability_L2(b, b_prev, snap.B),
                se=spectral_efficiency_sum(r, snap.B),
            )
            rows.append(row)
            b_prev = b
    return pd.DataFrame(rows)


def bootstrap_ci(
    x: np.ndarray, iters: int = 1000, alpha: float = 0.05, rng: np.random.Generator | None = None
):
    rng = rng or np.random.default_rng(0)
    means = []
    n = len(x)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        means.append(np.mean(x[idx]))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi

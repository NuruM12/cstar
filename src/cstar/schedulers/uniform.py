from __future__ import annotations

import numpy as np

from ..optimization.projection import project_capped_simplex
from .base import Scheduler


class Uniform(Scheduler):
    """
    Equal-share bandwidth allocator with per-user caps.

    Rules:
    - Split total B equally among *serviceable* users (capacity > 0).
    - Enforce per-user caps; any leftover is re-distributed uniformly to those
      still under cap (via capped-simplex projection).
    - Caps default to demand/capacity. You can override with ctx["caps"].
    - If demand.sum() == 0, we treat caps as infinite (except for users with c<=0).

    Notes:
    - Users with non-positive capacity receive zero bandwidth.
    - If total caps < B, sum(b) will equal sum(caps); otherwise it will equal B.
    """

    def allocate(self, demand: np.ndarray, capacity: np.ndarray, B: float, ctx):
        d = np.asarray(demand, dtype=float)
        c = np.asarray(capacity, dtype=float)
        assert d.ndim == 1 and c.ndim == 1 and d.shape == c.shape, "shape mismatch"
        n = d.size
        B = float(B)
        ctx = ctx or {}

        b0 = np.zeros(n, dtype=float)
        if n == 0 or B <= 0.0:
            return b0

        # Serviceable users (positive spectral efficiency)
        mask = c > 0.0
        k = int(mask.sum())
        if k == 0:
            return b0

        # Default caps: demand / capacity; allow override; fallback if all demand==0
        if "caps" in ctx:
            caps = np.asarray(ctx["caps"], dtype=float)
            assert caps.shape == d.shape, "ctx['caps'] shape mismatch"
        else:
            if float(d.sum()) <= 0.0:
                caps = np.where(mask, np.inf, 0.0)
            else:
                caps = np.divide(d, c, out=np.full(n, np.inf), where=mask)

        # Equal split over serviceable users, then project to capped simplex
        equal = B / k
        b0 = np.where(mask, equal, 0.0)

        b = project_capped_simplex(b0, caps, B)
        # Guards
        b = np.where(mask, b, 0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        return b

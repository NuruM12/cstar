from __future__ import annotations

import numpy as np

from .base import Scheduler


class GreedySINR(Scheduler):
    """
    Greedy throughput maximizer under per-user bandwidth caps.

    Strategy:
    - Sort users by spectral efficiency (capacity) descending.
    - Allocate each up to its cap, then move to the next, until budget runs out.
    - Users with non-positive capacity are not served.
    - Caps default to demand/capacity; you may override with ctx["caps"].

    Note:
    - If total caps < B, the allocation will sum to sum(caps); otherwise to B.
      In tests we choose cases with sum(caps) >= B.
    """

    def allocate(self, demand: np.ndarray, capacity: np.ndarray, B: float, ctx):
        d = np.asarray(demand, dtype=float).copy()
        c = np.asarray(capacity, dtype=float).copy()
        assert d.ndim == 1 and c.ndim == 1 and d.shape == c.shape, "shape mismatch"
        n = d.size
        B = float(B)
        ctx = ctx or {}

        b = np.zeros(n, dtype=float)
        if n == 0 or B <= 0.0:
            return b

        mask = c > 0.0

        # Caps: default demand/capacity, allow explicit override via ctx["caps"]
        if "caps" in ctx:
            caps = np.asarray(ctx["caps"], dtype=float)
            assert caps.shape == d.shape, "ctx['caps'] shape mismatch"
        else:
            caps = np.divide(d, c, out=np.full(n, np.inf), where=c > 0.0)

        # Non-serviceable users cannot get bandwidth
        caps = np.where(mask, caps, 0.0)

        # Greedy by capacity (descending)
        order = np.argsort(-c)  # indices sorted by descending capacity
        remaining = B
        for i in order:
            if not mask[i] or remaining <= 0.0:
                continue
            cap_i = caps[i]
            # If cap is infinite, take all remaining
            take = remaining if not np.isfinite(cap_i) else min(cap_i, remaining)
            if take > 0.0:
                b[i] = take
                remaining -= take

        # Final guards
        b = np.where(mask, b, 0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        return b

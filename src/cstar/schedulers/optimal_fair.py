from __future__ import annotations

import numpy as np

from ..optimization.projection import project_capped_simplex
from .base import Scheduler


class OptimalFair(Scheduler):
    r"""
    Max–min *rate* fairness with per-user caps.

    We find the largest common rate r* s.t. for each serviceable user i (capacity c_i > 0):
        b_i = min(cap_i, r* / c_i)
    and \sum_i b_i <= B, 0 <= b_i.
    - cap_i defaults to demand_i / c_i (how much bandwidth to satisfy demand).
      Override with ctx["caps"] if desired.
    - Users with non-positive capacity receive zero bandwidth.
    - If total caps < B, all caps are saturated and sum(b) = sum(caps).

    Numerical guards ensure stability; a final projection enforces exact caps and sum.
    """

    def allocate(self, demand: np.ndarray, capacity: np.ndarray, B: float, ctx):
        d = np.asarray(demand, dtype=float)
        c = np.asarray(capacity, dtype=float)
        assert d.ndim == 1 and c.ndim == 1 and d.shape == c.shape, "shape mismatch"
        n = d.size
        B = float(B)
        ctx = ctx or {}

        if n == 0 or B <= 0.0:
            return np.zeros(n, dtype=float)

        # Serviceable mask
        mask = c > 0.0
        if not mask.any():
            return np.zeros(n, dtype=float)

        # Caps
        if "caps" in ctx:
            caps = np.asarray(ctx["caps"], dtype=float)
            assert caps.shape == d.shape, "ctx['caps'] shape mismatch"
        else:
            if float(d.sum()) <= 0.0:
                # No demand → effectively unbounded (for serviceable users)
                caps = np.where(mask, np.inf, 0.0)
            else:
                caps = np.divide(d, c, out=np.full(n, np.inf), where=mask)

        # If the total feasible cap mass is already below B, saturate caps.
        # Treat inf as very large for the sum test.
        caps_finite = np.where(np.isfinite(caps), caps, 0.0)
        sum_caps = float(caps_finite[mask].sum())
        if sum_caps <= B + 1e-9:
            b = np.where(mask, caps_finite, 0.0)
            # Final projection (will keep b == caps_finite if sum_caps <= B)
            return project_capped_simplex(b, caps, B)

        # Otherwise: bisection on the target *rate* r so that sum min(caps_i, r/c_i) == B
        c_safe = c + 1e-12

        def bw_at_rate(r: float) -> float:
            # For each serviceable i: needed bandwidth to reach rate r is r/c_i,
            # but limited by cap_i.
            need = np.minimum(caps[mask], r / c_safe[mask])
            return float(np.sum(need))

        # Bracket r*
        lo = 0.0
        # Upper bound: at least the largest single-user demand rate,
        # and grow until total >= B.
        # A simple heuristic start:
        hi = float(np.nanmax(d[mask])) if np.isfinite(d[mask]).any() else 1.0
        hi = max(1.0, hi)

        # Expand hi until enough total bandwidth is required to reach hi
        for _ in range(40):
            if bw_at_rate(hi) >= B:
                break
            hi *= 2.0

        # Bisection
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if bw_at_rate(mid) >= B:
                hi = mid
            else:
                lo = mid

        r_star = 0.5 * (lo + hi)

        # Construct allocation at r_star and project for exactness
        b = np.zeros(n, dtype=float)
        b[mask] = np.minimum(caps[mask], r_star / c_safe[mask])

        # Enforce exact simplex with caps (handles roundoff and any inf caps)
        b = project_capped_simplex(b, caps, B)

        # Guards
        b = np.where(mask, b, 0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        return b

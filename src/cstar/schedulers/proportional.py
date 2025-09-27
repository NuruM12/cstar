from __future__ import annotations

import numpy as np

from ..optimization.projection import project_capped_simplex
from .base import Scheduler


class Proportional(Scheduler):
    """
    Allocate bandwidth proportional to demand^p over users with positive capacity.

    Behavior:
    - Caps (b_i <= d_i / c_i) are used only when they BIND (sum caps > B).
    - You may OVERRIDE caps via ctx["caps"] (array-like, same length as demand).
    - Users with c_i <= 0 get zero bandwidth and cannot receive allocation.
    - Exact sum=B is enforced via capped-simplex projection.

    Under binding caps, we solve a weighted water-filling:
        b_i = min(cap_i, λ * u_i),  where u_i = demand_i^p (on c_i>0),
    choosing λ so that sum_i b_i = B. This saturates tight caps first, then
    redistributes remaining budget across non-saturated users by weights u_i.
    """

    def allocate(self, demand: np.ndarray, capacity: np.ndarray, B: float, ctx):
        d = np.asarray(demand, dtype=float).copy()
        c = np.asarray(capacity, dtype=float).copy()
        assert d.ndim == 1 and c.ndim == 1 and d.shape == c.shape, "shape mismatch"
        B = float(B)
        n = d.size
        ctx = ctx or {}

        mask = c > 0.0  # only these users are serviceable

        # --- caps: demand/capacity by default; allow override via ctx["caps"] ---
        if "caps" in ctx:
            caps_in = np.asarray(ctx["caps"], dtype=float)
            assert caps_in.shape == d.shape, "ctx['caps'] shape mismatch"
            raw_caps = caps_in.copy()
        else:
            raw_caps = np.divide(d, c, out=np.full(n, np.inf), where=c > 0)

        # Non-serviceable users cannot receive bandwidth
        raw_caps = np.where(mask, raw_caps, 0.0)

        # Decide if caps bind
        finite_caps = np.where(np.isfinite(raw_caps), raw_caps, 0.0)
        sum_caps = float(finite_caps.sum())
        caps_bind = sum_caps > B + 1e-9

        # Demand-powered weights
        p = float(ctx.get("demand_power", 1.0))
        u = np.zeros(n, dtype=float)
        u[mask] = np.clip(d[mask], 0.0, None) ** p

        if not caps_bind:
            # Non-binding caps: ignore caps (except blocking c<=0) and split by weights
            total_u = u[mask].sum()
            b0 = np.zeros(n, dtype=float)
            if total_u > 0.0:
                b0[mask] = B * (u[mask] / total_u)
            else:
                m = int(mask.sum())
                if m == 0:
                    return np.zeros(n, dtype=float)
                b0[mask] = B / m
            caps = np.where(mask, np.inf, 0.0)
            return project_capped_simplex(b0, caps, B)

        # Binding caps: weighted water-filling with caps
        caps = raw_caps

        # If all positive-capacity weights are zero, fall back to equal weights
        if u[mask].sum() == 0.0:
            u[mask] = 1.0

        def total_bw(lam: float) -> float:
            return float(np.minimum(caps, lam * u).sum())

        # Find λ by bisection so that sum min(caps, λ u) = B
        lo, hi = 0.0, 1.0
        # Expand hi until feasible (or until all caps saturate, which we know sum > B)
        for _ in range(60):
            if total_bw(hi) >= B:
                break
            hi *= 2.0

        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if total_bw(mid) >= B:
                hi = mid
            else:
                lo = mid

        lam = 0.5 * (lo + hi)
        b = np.minimum(caps, lam * u)

        # Final projection to nail exact sum=B and preserve caps
        return project_capped_simplex(b, caps, B)

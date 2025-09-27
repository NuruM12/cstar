import numpy as np

from ..optimization.projection import project_capped_simplex
from .base import Scheduler


class WRR(Scheduler):
    """
    Weighted Round Robin allocator with per-user caps.

    • Accepts optional ctx["weights"] (nonnegative priorities). If omitted,
      builds weights as a convex combo of normalized demand & capacity:
          w = λ·(d/∑d) + (1-λ)·(c/∑c), with λ=ctx.get("lambda", 0.6).
    • Enforces caps b_i ≤ demand_i / capacity_i (0 if c_i=0).
    • Redistributes leftover bandwidth among non-saturated users
      proportionally to their weights (round-robin flavor).
    • Finishes with a capped-simplex projection for exact sum=B and safety.
    """

    def allocate(self, demand: np.ndarray, capacity: np.ndarray, B: float, ctx):
        eps = 1e-12
        demand = np.asarray(demand, dtype=float)
        capacity = np.asarray(capacity, dtype=float)
        B = float(B)

        n = demand.size
        if n == 0 or B <= 0.0:
            return np.zeros_like(demand, dtype=float)

        # Eligible users (nonzero capacity)
        eligible = capacity > eps

        # Per-user caps from demand constraints; 0 for ineligible users
        caps = np.full(n, 0.0, dtype=float)
        caps[eligible] = demand[eligible] / (capacity[eligible] + eps)

        # Build weights
        w = ctx.get("weights", None)
        if w is None:
            lam = float(ctx.get("lambda", 0.6))
            d_norm = demand / (demand.sum() + eps)
            c_norm = capacity / (capacity.sum() + eps)
            w = lam * d_norm + (1.0 - lam) * c_norm
        w = np.asarray(w, dtype=float)
        w = np.clip(w, 0.0, np.inf)

        # Users with zero capacity should not receive bandwidth
        w = np.where(eligible, w, 0.0)

        # If all weights zero, fall back to uniform over eligible users
        if w.sum() <= eps:
            w = np.where(eligible, 1.0, 0.0)

        # Iterative weighted redistribution respecting caps
        b = np.zeros(n, dtype=float)
        remaining = B
        active = (w > eps) & (caps > eps)

        # Guard: if total caps < B, spend what you can and project for exactness.
        total_cap = float(np.where(np.isfinite(caps), caps, 0.0).sum())
        if total_cap <= B + 1e-9:
            b = np.minimum(caps, np.where(active, np.inf, 0.0))
            return project_capped_simplex(b, caps, B)

        # Distribute in passes, deactivating saturated users each pass.
        # (Converges rapidly because at least one user saturates or budget is spent.)
        for _ in range(128):
            if remaining <= 1e-9 or not np.any(active):
                break
            w_active = np.where(active, w, 0.0)
            s = w_active.sum()
            if s <= eps:
                break
            share = remaining * (w_active / s)
            # Apply share but clamp to caps
            new_b = np.minimum(b + share, caps)
            gained = float((new_b - b).sum())
            b = new_b
            remaining = max(0.0, B - float(b.sum()))
            # Deactivate anyone who hit their cap (with a small tolerance)
            active &= b < caps - 1e-9
            if gained <= 1e-12:
                break

        # Final safety: exact sum=B and all constraints via projection
        return project_capped_simplex(b, caps, B)

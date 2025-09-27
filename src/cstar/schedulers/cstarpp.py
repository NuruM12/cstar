import numpy as np

from ..optimization.projection import project_capped_simplex
from .base import Scheduler
from .optimal_fair import OptimalFair
from .proportional import Proportional
from .uniform import Uniform
from .wrr import WRR


class CStarPP(Scheduler):
    """
    Composite C-STAR++ allocator: convex blend of Uniform / Proportional / WRR / OptimalFair.
    The gating looks at demand skew, capacity variance, and simple traffic dynamics hints
    (if provided in ctx) to choose weights, then we project onto the capped simplex.
    """

    def __init__(self):
        self._u = Uniform()
        self._p = Proportional()
        self._w = WRR()
        self._f = OptimalFair()

    @staticmethod
    def _safe_caps(demand: np.ndarray, capacity: np.ndarray) -> np.ndarray:
        # cap in bandwidth units: b_i <= demand_i / capacity_i (if capacity_i>0), else 0
        caps = np.divide(
            demand,
            capacity,
            out=np.full_like(demand, np.inf, dtype=float),
            where=capacity > 0.0,
        )
        caps = np.where(capacity > 0.0, caps, 0.0)
        return caps

    @staticmethod
    def _gini(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return 0.0
        mu = float(x.mean())
        if mu <= 0:
            return 0.0
        diffs = np.abs(x[:, None] - x[None, :]).mean()
        return float(diffs / (2.0 * mu + 1e-12))

    def _gate(self, demand: np.ndarray, capacity: np.ndarray, ctx: dict) -> np.ndarray:
        # Manual override: ctx["weights"] is [wU, wP, wW, wF]
        if "weights" in ctx and ctx["weights"] is not None:
            w = np.asarray(ctx["weights"], dtype=float)
            if w.size != 4:
                raise ValueError("weights must be a length-4 vector [wU,wP,wW,wF]")
            w = np.clip(w, 0.0, None)
            s = w.sum()
            return w / s if s > 0 else np.array([0.25, 0.25, 0.25, 0.25])

        # Baseline blend (slightly fairness-leaning):
        wU, wP, wW, wF = 0.15, 0.25, 0.25, 0.35

        # Signals
        gini = self._gini(demand)
        varC = float(np.var(capacity / (capacity.mean() + 1e-12)))
        ratio_minmax = float(demand.max() / max(demand.min(), 1e-12))

        # If capacities are very non-uniform, give WRR a bit more sway (but keep fairness)
        if varC > 0.6:
            bump = min(0.15, 0.5 * (varC - 0.6))
            wW += bump
            # take proportionally from U and P to avoid hurting fairness too much
            take = bump
            tU = min(take * 0.6, wU - 0.05)
            tP = min(take * 0.4, wP - 0.05)
            wU -= max(0.0, tU)
            wP -= max(0.0, tP)

        # If demand distribution is moderately skewed, lean toward fairness.
        if gini > 0.35 and varC < 0.4:
            add = min(0.15, 0.5 * (gini - 0.35 + 1e-9))
            wF += add
            # take from WRR first, then P if needed
            tW = min(add * 0.6, max(0.0, wW - 0.08))
            wW -= tW
            add -= tW
            if add > 0:
                tP = min(add, max(0.0, wP - 0.10))
                wP -= tP
                add -= tP

        # Mild dynamics hooks (optional hints)
        mobility = float(ctx.get("mobility_index", 0.0))  # 0..1
        anomaly = float(ctx.get("traffic_anomaly", 0.0))  # 0..1
        dyn = np.clip(0.5 * mobility + 0.5 * anomaly, 0.0, 1.0)
        if dyn > 0:
            nud = 0.15 * dyn
            wF += nud
            wU += 0.40 * nud
            wW -= 0.80 * nud

        # **Hard fairness guardrails under extreme demand skew & similar capacities**
        # If one (or a few) users are tiny vs others and capacities are ~equal, ensure
        # the composite is clearly more fairness-leaning than a WRR-heavy mix.
        # This directly covers cases like [1,100,100,100] with equal capacities.
        if ratio_minmax > 50.0 and varC < 0.25:
            # Cap WRR further and ensure a strong OptimalFair presence
            wW = min(wW, 0.10)
            need = max(0.0, 0.75 - wF)  # target >= 0.75 fairness weight
            takeP = min(need, max(0.0, wP - 0.05))
            wP -= takeP
            wF += takeP
            need -= takeP
            if need > 0:
                takeU = min(need, max(0.0, wU - 0.05))
                wU -= takeU
                wF += takeU

        w = np.array([wU, wP, wW, wF], dtype=float)
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s <= 0:
            w[:] = 0.25
            s = 1.0
        return w / s

    def allocate(self, demand: np.ndarray, capacity: np.ndarray, B: float, ctx: dict):
        demand = np.asarray(demand, dtype=float).copy()
        capacity = np.asarray(capacity, dtype=float).copy()
        n = demand.size
        assert capacity.shape == demand.shape == (n,)

        caps = self._safe_caps(demand, capacity)

        lam = float(ctx.get("lambda", 0.6))
        bU = self._u.allocate(demand, capacity, B, ctx={})
        bP = self._p.allocate(demand, capacity, B, ctx={})
        bW = self._w.allocate(demand, capacity, B, ctx={"lambda": lam})
        bF = self._f.allocate(demand, capacity, B, ctx={})

        w = self._gate(demand, capacity, ctx)  # [wU,wP,wW,wF]
        b = w[0] * bU + w[1] * bP + w[2] * bW + w[3] * bF

        # Final projection: honor caps and exact budget
        return project_capped_simplex(b, caps, float(B))

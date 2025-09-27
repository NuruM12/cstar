from __future__ import annotations

import numpy as np


def jain_fairness(r: np.ndarray, eps: float = 1e-12) -> float:
    r = np.asarray(r, float)
    s1 = float(r.sum())
    s2 = float((r**2).sum())
    n = r.size
    return (s1 * s1) / (n * s2 + eps) if s2 > 0 else 0.0


def throughput_sum(r: np.ndarray) -> float:
    return float(np.asarray(r, float).sum())


def unmet_demand_U(r: np.ndarray, d: np.ndarray, eps: float = 1e-12) -> float:
    """1 - normalized shortfall (higher is better)."""
    r = np.asarray(r, float)
    d = np.asarray(d, float)
    short = np.clip(d - r, 0.0, None).sum()
    denom = d.sum() + eps
    return 1.0 - (short / denom if denom > 0 else 0.0)


def cvar_shortfall(r: np.ndarray, d: np.ndarray, alpha: float = 0.8, eps: float = 1e-12) -> float:
    """Return 1 - CVaR_alpha of shortfall normalized by mean demand (higher is better)."""
    r = np.asarray(r, float)
    d = np.asarray(d, float)
    s = np.clip(d - r, 0.0, None)
    if s.size == 0:
        return 1.0
    q = np.quantile(s, alpha)
    tail = s[s >= q]
    cvar = float(tail.mean()) if tail.size else 0.0
    denom = float(d.mean()) + eps
    return 1.0 - cvar / denom


def entropy_alloc(bw: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized allocation entropy."""
    b = np.asarray(bw, float)
    Z = b.sum() + eps
    p = b / Z
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    H = float(-(p * np.log(p)).sum())
    import math

    return H / (math.log(bw.size) + eps)


def stability_L2(
    b_now: np.ndarray, b_prev: np.ndarray | None, B: float, eps: float = 1e-12
) -> float:
    if b_prev is None:
        return 1.0
    diff = np.linalg.norm(np.asarray(b_now, float) - np.asarray(b_prev, float))
    return 1.0 - (diff * diff) / (B * B + eps)


def spectral_efficiency_sum(r: np.ndarray, B: float, eps: float = 1e-12) -> float:
    return float(np.asarray(r, float).sum() / (B + eps))

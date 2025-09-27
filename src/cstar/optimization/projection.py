import numpy as np


def project_capped_simplex(
    b0: np.ndarray, caps: np.ndarray, B: float, *, tol: float = 1e-9, max_iter: int = 12
) -> np.ndarray:
    b = np.clip(np.asarray(b0, float), 0.0, np.asarray(caps, float))
    if b.sum() <= tol:
        idx = np.isfinite(caps) & (caps > 0)
        if not np.any(idx):
            return b
        share = B / idx.sum()
        b[idx] = np.minimum(caps[idx], share)
    for _ in range(max_iter):
        s = b.sum()
        if abs(s - B) <= tol:
            break
        if s <= tol:
            idx = np.isfinite(caps) & (caps > 0)
            if not np.any(idx):
                return b
            share = B / idx.sum()
            b[idx] = np.minimum(caps[idx], share)
            continue
        b = np.clip(b * (B / s), 0.0, caps)
    diff = B - b.sum()
    if abs(diff) > tol:
        free = b < (caps - 1e-12)
        if np.any(free):
            b[free] += diff / free.sum()
            b = np.clip(b, 0.0, caps)
    return b

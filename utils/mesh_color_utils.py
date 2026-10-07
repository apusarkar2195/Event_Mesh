"""Shared mesh color smoothing utilities."""
import numpy as np


def smooth_vertex_colors(
    colors: np.ndarray,
    tris: np.ndarray,
    method: str = "mean",
    iterations: int = 1,
    huber_quantile: float = 0.1,
) -> np.ndarray:
    """
    Smooth vertex colors using mesh connectivity.
    colors: [V, 3] float in [0, 1]
    method: 'mean' (Laplacian, fast), 'median' (robust, slower), 'huber' (trimmed mean)
    """
    from scipy import sparse

    V = colors.shape[0]
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    rows = np.concatenate([a, b, c, b, c, a])
    cols = np.concatenate([b, a, a, c, b, c])
    data = np.ones(len(rows), dtype=np.float64)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(V, V))
    A = (A + sparse.identity(V)).tocsr()
    deg = np.array(A.sum(axis=1)).ravel()
    deg_inv = np.where(deg > 0, 1.0 / deg, 1.0)
    A = A.multiply(deg_inv[:, np.newaxis]).tocsr()

    out = colors.astype(np.float64)
    for _ in range(iterations):
        if method == "mean":
            out = A @ out
        elif method == "median":
            out_next = out.copy()
            for v in range(V):
                ind = A[v].indices
                if len(ind) > 0:
                    pool = out[ind]
                    out_next[v] = np.median(pool, axis=0)
            out = out_next
        elif method == "huber":
            out_next = out.copy()
            for v in range(V):
                ind = A[v].indices
                if len(ind) > 0:
                    pool = out[ind]
                    q_lo, q_hi = np.percentile(pool, [huber_quantile * 100, (1 - huber_quantile) * 100], axis=0)
                    mask = (pool >= q_lo) & (pool <= q_hi)
                    mask = mask.all(axis=1)
                    if mask.sum() > 0:
                        out_next[v] = pool[mask].mean(axis=0)
            out = out_next
        else:
            raise ValueError(f"Unknown method: {method}")

    return np.clip(out, 0, 1).astype(np.float32)

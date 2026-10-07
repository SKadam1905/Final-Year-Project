"""Ground-truth density maps from head-point annotations.

A density map has the image's size and sums to the number of annotated heads, so summing
it over any region gives the head count there. Each head is a normalised Gaussian whose
width adapts to local crowding (geometry-adaptive kernels, MCNN - Zhang et al., CVPR 2016):

    sigma_i = beta * mean distance from head i to its k nearest heads   (beta=0.3, k=3)

Each kernel is renormalised after clipping at the image border, so heads near the edge
still contribute exactly 1 to the count.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def clamp_points(points, width: int, height: int) -> np.ndarray:
    """Clamp (x, y) points into the image; some annotations sit a pixel outside."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    eps = 1e-6
    return np.column_stack([np.clip(pts[:, 0], 0, width - eps), np.clip(pts[:, 1], 0, height - eps)])


def adaptive_sigmas(
    points: np.ndarray,
    k: int = 3,
    beta: float = 0.3,
    fallback: float = 15.0,
    min_sigma: float = 1.0,
    max_sigma: float = 100.0,
) -> np.ndarray:
    n = len(points)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.full(1, fallback)
    neighbours = min(k, n - 1)
    dists, _ = cKDTree(points).query(points, k=neighbours + 1)  # column 0 is the point itself
    return np.clip(beta * dists[:, 1:].mean(axis=1), min_sigma, max_sigma)


def density_map(
    points,
    shape: tuple[int, int],
    sigma: float | None = None,
    truncate: float = 3.0,
    **sigma_kwargs,
) -> np.ndarray:
    """Density map of `shape` (h, w); adaptive kernels unless a fixed `sigma` is given."""
    height, width = shape
    pts = clamp_points(points, width, height)
    dmap = np.zeros((height, width), np.float64)
    if len(pts) == 0:
        return dmap.astype(np.float32)
    if sigma is None:
        sigmas = adaptive_sigmas(pts, **sigma_kwargs)
    elif sigma > 0:
        sigmas = np.full(len(pts), float(sigma))
    else:
        raise ValueError("sigma must be positive")

    for (x, y), s in zip(pts, sigmas):
        r = max(1, int(np.ceil(truncate * s)))
        cx, cy = int(x), int(y)
        x0, x1 = max(cx - r, 0), min(cx + r + 1, width)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, height)
        gx = np.exp(-0.5 * ((np.arange(x0, x1) + 0.5 - x) / s) ** 2)
        gy = np.exp(-0.5 * ((np.arange(y0, y1) + 0.5 - y) / s) ** 2)
        kernel = np.outer(gy, gx)
        dmap[y0:y1, x0:x1] += kernel / kernel.sum()
    return dmap.astype(np.float32)


def sum_pool(density: np.ndarray, factor: int = 8) -> np.ndarray:
    """Downsample by summing factor x factor blocks; preserves the total exactly."""
    h, w = density.shape
    if h % factor or w % factor:
        raise ValueError(f"density map {w}x{h} is not a multiple of {factor}")
    blocks = density.reshape(h // factor, factor, w // factor, factor)
    return blocks.sum(axis=(1, 3), dtype=np.float64).astype(np.float32)

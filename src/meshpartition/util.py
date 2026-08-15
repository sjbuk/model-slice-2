"""Small deterministic helpers shared across pipeline stages."""

from __future__ import annotations

import numpy as np


class UnionFind:
    """Disjoint-set over integers 0..n-1, path-compressed, union by stable min-index rule.

    Union order never affects the resulting partition, only bookkeeping,
    so results are deterministic regardless of the order pairs are unioned in.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Deterministic: the smaller root index always wins as the new root.
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def triangle_areas(corner_positions: np.ndarray) -> np.ndarray:
    """corner_positions: (F, 3, 3) -> (F,) triangle areas."""
    v0, v1, v2 = corner_positions[:, 0], corner_positions[:, 1], corner_positions[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def triangle_centroids(corner_positions: np.ndarray) -> np.ndarray:
    return corner_positions.mean(axis=1)


def bbox_diagonal(positions: np.ndarray) -> float:
    if len(positions) == 0:
        return 0.0
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def face_normals(corner_positions: np.ndarray) -> np.ndarray:
    """corner_positions: (F, 3, 3) -> (F, 3) unit geometric normals.

    Derived from triangle winding, not any authored normal attribute --
    Stage 2's geometric reasoning (normal agreement, winding number) needs a
    normal for every face regardless of whether the source mesh carries one.
    """
    v0, v1, v2 = corner_positions[:, 0], corner_positions[:, 1], corner_positions[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(cross, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-300, 1.0, lengths)
    return cross / lengths


def closest_points_on_triangles(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest point (and distance) from p to each of T triangles (a[i],b[i],c[i]).

    Standard region-based algorithm (Ericson, "Real-Time Collision Detection",
    5.1.5), vectorized over all T triangles at once.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)

    t = len(a)
    result = np.empty((t, 3), dtype=np.float64)
    unresolved = np.ones(t, dtype=bool)

    mask = (d1 <= 0) & (d2 <= 0) & unresolved
    result[mask] = a[mask]
    unresolved &= ~mask

    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    mask = (d3 >= 0) & (d4 <= d3) & unresolved
    result[mask] = b[mask]
    unresolved &= ~mask

    vc = d1 * d4 - d3 * d2
    mask = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & unresolved
    denom = np.where(mask, d1 - d3, 1.0)
    v = np.where(mask, d1 / denom, 0.0)
    result[mask] = (a + ab * v[:, None])[mask]
    unresolved &= ~mask

    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    mask = (d6 >= 0) & (d5 <= d6) & unresolved
    result[mask] = c[mask]
    unresolved &= ~mask

    vb = d5 * d2 - d1 * d6
    mask = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & unresolved
    denom = np.where(mask, d2 - d6, 1.0)
    w = np.where(mask, d2 / denom, 0.0)
    result[mask] = (a + ac * w[:, None])[mask]
    unresolved &= ~mask

    va = d3 * d6 - d5 * d4
    mask = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & unresolved
    bc_denom = np.where(mask, (d4 - d3) + (d5 - d6), 1.0)
    w_bc = np.where(mask, (d4 - d3) / bc_denom, 0.0)
    result[mask] = (b + (c - b) * w_bc[:, None])[mask]
    unresolved &= ~mask

    # Remaining: interior of the face.
    denom = np.where(unresolved, va + vb + vc, 1.0)
    v = np.where(unresolved, vb / denom, 0.0)
    w = np.where(unresolved, vc / denom, 0.0)
    result[unresolved] = (a + ab * v[:, None] + ac * w[:, None])[unresolved]

    dists = np.linalg.norm(result - p, axis=1)
    return result, dists


def generalized_winding_number(p: np.ndarray, positions: np.ndarray, faces: np.ndarray) -> float:
    """Generalized winding number of p w.r.t. the (possibly non-watertight)
    mesh (positions, faces). ~1 fully enclosed, ~0 fully outside, fractional
    for partial/open surfaces (Barill et al. 2018 / Jacobson et al. 2013).

    Brute-force O(F) sum, not tree-accelerated -- fine at prototype/test scale
    (production would use libigl's fast_winding_number with a BVH).
    """
    if len(faces) == 0:
        return 0.0
    tri = positions[faces]
    a = tri[:, 0] - p
    b = tri[:, 1] - p
    c = tri[:, 2] - p
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    lc = np.linalg.norm(c, axis=1)

    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denom = (
        la * lb * lc
        + np.einsum("ij,ij->i", a, b) * lc
        + np.einsum("ij,ij->i", b, c) * la
        + np.einsum("ij,ij->i", c, a) * lb
    )
    solid_angle = 2.0 * np.arctan2(numerator, denom)
    return float(np.sum(solid_angle) / (4.0 * np.pi))

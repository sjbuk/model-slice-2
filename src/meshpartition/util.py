"""Small deterministic helpers shared across pipeline stages."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


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


@njit(cache=True, parallel=True)
def closest_point_distance_matrix(
    query: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """(S,3) query points x (T,3)/(T,3)/(T,3) triangle corners -> (S, T)
    closest-point distances.

    Standard region-based algorithm (Ericson, "Real-Time Collision
    Detection", 5.1.5), one query/triangle pair at a time. This used to be
    vectorized through numpy instead of compiled, which allocates ~15-20
    T-length temporary arrays per call; once S and T are both in the
    thousands (a real hard-surface asset's bridge search can have S*T in
    the tens of millions per satellite), that allocation traffic dominates
    over the actual arithmetic. A compiled scalar loop needs none of it --
    just the (S, T) output -- and parallelizes over query points.
    """
    s = query.shape[0]
    t = a.shape[0]
    dists = np.empty((s, t), dtype=np.float64)
    for si in prange(s):
        px = query[si, 0]
        py = query[si, 1]
        pz = query[si, 2]
        for ti in range(t):
            ax, ay, az = a[ti, 0], a[ti, 1], a[ti, 2]
            bx, by, bz = b[ti, 0], b[ti, 1], b[ti, 2]
            cx, cy, cz = c[ti, 0], c[ti, 1], c[ti, 2]

            abx, aby, abz = bx - ax, by - ay, bz - az
            acx, acy, acz = cx - ax, cy - ay, cz - az
            apx, apy, apz = px - ax, py - ay, pz - az
            d1 = abx * apx + aby * apy + abz * apz
            d2 = acx * apx + acy * apy + acz * apz

            if d1 <= 0.0 and d2 <= 0.0:
                rx, ry, rz = ax, ay, az
            else:
                bpx, bpy, bpz = px - bx, py - by, pz - bz
                d3 = abx * bpx + aby * bpy + abz * bpz
                d4 = acx * bpx + acy * bpy + acz * bpz
                if d3 >= 0.0 and d4 <= d3:
                    rx, ry, rz = bx, by, bz
                else:
                    vc = d1 * d4 - d3 * d2
                    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                        v = d1 / (d1 - d3)
                        rx, ry, rz = ax + abx * v, ay + aby * v, az + abz * v
                    else:
                        cpx, cpy, cpz = px - cx, py - cy, pz - cz
                        d5 = abx * cpx + aby * cpy + abz * cpz
                        d6 = acx * cpx + acy * cpy + acz * cpz
                        if d6 >= 0.0 and d5 <= d6:
                            rx, ry, rz = cx, cy, cz
                        else:
                            vb = d5 * d2 - d1 * d6
                            if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                                w = d2 / (d2 - d6)
                                rx, ry, rz = ax + acx * w, ay + acy * w, az + acz * w
                            else:
                                va = d3 * d6 - d5 * d4
                                if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                                    w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                                    rx = bx + (cx - bx) * w_bc
                                    ry = by + (cy - by) * w_bc
                                    rz = bz + (cz - bz) * w_bc
                                else:
                                    denom = va + vb + vc
                                    v = vb / denom
                                    w = vc / denom
                                    rx = ax + abx * v + acx * w
                                    ry = ay + aby * v + acy * w
                                    rz = az + abz * v + acz * w

            dx, dy, dz = rx - px, ry - py, rz - pz
            dists[si, ti] = (dx * dx + dy * dy + dz * dz) ** 0.5
    return dists


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

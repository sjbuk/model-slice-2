"""Stage 7 -- Repair (Atomic Dump).

    1. Satellite fixup: force every satellite's cells to its anchor's label.
    2. Connectivity ("the dump"): compute connected components of each label
       over the FULL dual graph (surface edges + bridges -- a satellite cell
       reassigned in step 1 is only "connected" to its host through the
       bridge edge that anchored it). Keep the largest component by area;
       donate every other ("orphan") component atomically to whichever
       neighboring region currently has the smallest area.
    3. Compactness: eigenvalue ratio of each region's face-centroid
       covariance. Regions over the elongation limit gamma get a raised
       alpha for the caller's next growth pass (Stage 8's relaxation loop).

Point 4 of the spec ("re-seed any emptied regions") is not implemented here:
under this construction a region can never actually end up empty. Step 1
never touches a region's own seed face (a seed is always chosen from major
territory in Stage 5, never satellite territory), and step 2's "keep the
largest component" always keeps at least one component -- the one containing
that seed, if no larger one exists under the same label. Emptying a region
would require Stage 8's later re-seed-at-medoid step to place a fresh seed
that then fails to claim anything during its own growth pass; that failure
mode belongs to Stage 8, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .mesh import WorkingMesh
from .stage1 import TriageResult
from .stage2 import BridgeResult
from .stage4 import DualGraph
from .stage6 import DEFAULT_ALPHA, GrowthResult
from .util import dual_graph_csr

DEFAULT_GAMMA = 2.5
ELONGATION_ALPHA_STEP = 0.5
ALPHA_MAX = 3.0


@dataclass
class RepairResult:
    label: np.ndarray  # (F,) int64, final region id per face after fixup + dump
    region_area: np.ndarray  # (R,) float64, area per region after repair
    elongation: np.ndarray  # (R,) float64, eigenvalue ratio rho_i per region
    next_alpha: np.ndarray  # (R,) float64, alpha_i for the next growth pass
    dumped: list[tuple[int, int, int]]  # (orphan_seed_face, from_label, to_label), in donation order


def _label_areas(working: WorkingMesh, label: np.ndarray, n_regions: int) -> np.ndarray:
    areas = np.zeros(n_regions, dtype=np.float64)
    for region_id in range(n_regions):
        areas[region_id] = float(working.face_areas[label == region_id].sum())
    return areas


@njit(cache=True, inline="always")
def _uf_find(parent: np.ndarray, x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


@njit(cache=True, inline="always")
def _uf_union(parent: np.ndarray, a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return
    # Deterministic, matching util.UnionFind: the smaller root index always
    # wins, so the resulting partition doesn't depend on edge-scan order.
    if ra < rb:
        parent[rb] = ra
    else:
        parent[ra] = rb


@njit(cache=True)
def _label_components_kernel(indptr: np.ndarray, nbr: np.ndarray, label: np.ndarray) -> np.ndarray:
    n = label.shape[0]
    parent = np.arange(n, dtype=np.int64)
    for face_id in range(n):
        for e in range(indptr[face_id], indptr[face_id + 1]):
            neighbor = nbr[e]
            if label[face_id] == label[neighbor]:
                _uf_union(parent, face_id, neighbor)

    roots = np.empty(n, dtype=np.int64)
    for face_id in range(n):
        roots[face_id] = _uf_find(parent, face_id)
    return roots


def label_components(
    working: WorkingMesh, dual_graph: DualGraph, label: np.ndarray
) -> dict[int, dict[int, list[int]]]:
    """Connected components of each label, under same-label adjacency in the
    full dual graph (including bridges). Returns label -> root -> face ids.

    The edge-scan + union-find pass is a numba kernel (own array-based
    union-find, not util.UnionFind -- that class is also used by Stage
    0/Stage 1 and this keeps their behavior untouched) over a cached CSR
    view of dual_graph.adjacency (see util.dual_graph_csr): this function is
    re-run by Stage 8's relax() on every Lloyd iteration, so the same
    per-run CSR cache that grow() uses applies here too.
    """
    indptr, nbr, _cost = dual_graph_csr(dual_graph)
    roots = _label_components_kernel(indptr, nbr, np.ascontiguousarray(label, dtype=np.int64))

    groups: dict[int, dict[int, list[int]]] = {}
    for face_id in range(working.face_count):
        lbl = int(label[face_id])
        root = int(roots[face_id])
        groups.setdefault(lbl, {}).setdefault(root, []).append(face_id)
    return groups


def apply_dump(
    working: WorkingMesh, dual_graph: DualGraph, label: np.ndarray, n_regions: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    label = label.copy()
    region_area = _label_areas(working, label, n_regions)
    dumped: list[tuple[int, int, int]] = []

    groups = label_components(working, dual_graph, label)
    for lbl in sorted(groups):
        islands = groups[lbl]
        if len(islands) <= 1:
            continue

        island_area = {root: sum(working.face_areas[f] for f in faces) for root, faces in islands.items()}
        keeper = max(islands, key=lambda r: (island_area[r], -min(islands[r])))
        orphan_roots = sorted((r for r in islands if r != keeper), key=lambda r: min(islands[r]))

        for root in orphan_roots:
            orphan_faces = islands[root]
            orphan_area = island_area[root]

            neighbor_labels = {
                int(label[neighbor])
                for f in orphan_faces
                for neighbor, _cost, _is_bridge in dual_graph.adjacency[f]
                if int(label[neighbor]) != lbl
            }
            if not neighbor_labels:
                continue  # unreachable: Stage 4 guarantees the dual graph is one connected component

            target = min(neighbor_labels, key=lambda r: (region_area[r], r))
            for f in orphan_faces:
                label[f] = target
            region_area[lbl] -= orphan_area
            region_area[target] += orphan_area
            dumped.append((min(orphan_faces), lbl, target))

    return label, region_area, dumped


def _elongation(working: WorkingMesh, label: np.ndarray, region_id: int) -> float:
    """Ratio of the length axis to the width axis of the region's face
    centroids -- eigenvalues[2] (length) over eigenvalues[1] (width), not
    eigenvalues[0] (thickness). A perfectly planar mesh (Fixture E, the
    Ribbon, has z=0 on every vertex) gives every region a near-zero variance
    in its thickness direction regardless of the region's actual shape, so
    dividing by the smallest eigenvalue would report near-infinite
    "elongation" even for a small, round patch -- it's measuring the mesh's
    flatness, not the region's. Length-over-width is what "ribbon-like"
    (long and narrow) actually means, and stays well-behaved for flat input.
    """
    centroids = working.face_centroids[label == region_id]
    if len(centroids) < 2:
        return 1.0
    centered = centroids - centroids.mean(axis=0)
    cov = (centered.T @ centered) / len(centroids)
    eigenvalues = np.linalg.eigvalsh(cov)  # ascending: thickness, width, length
    length = eigenvalues[2]
    width = max(eigenvalues[1], length * 1e-9, 1e-300)
    return float(length / width)


def repair(
    working: WorkingMesh,
    dual_graph: DualGraph,
    triage: TriageResult,
    bridges: BridgeResult,
    growth: GrowthResult,
    alpha: float | np.ndarray = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
) -> RepairResult:
    n_regions = len(growth.region_seed)
    label = growth.label.copy()

    for satellite, anchor in bridges.anchors.items():
        anchor_label = int(label[anchor.other_face])
        satellite_faces = triage.face_component_id == satellite
        label[satellite_faces] = anchor_label

    label, region_area, dumped = apply_dump(working, dual_graph, label, n_regions)

    # alpha may be a single value (Stage 6/7 run standalone) or one value per
    # region (Stage 8 carrying each region's own already-raised alpha forward).
    alpha_per_region = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (n_regions,))

    elongation = np.ones(n_regions, dtype=np.float64)
    next_alpha = alpha_per_region.copy()
    for region_id in range(n_regions):
        rho = _elongation(working, label, region_id)
        elongation[region_id] = rho
        if rho > gamma:
            next_alpha[region_id] = min(alpha_per_region[region_id] + ELONGATION_ALPHA_STEP, ALPHA_MAX)

    return RepairResult(
        label=label,
        region_area=region_area,
        elongation=elongation,
        next_alpha=next_alpha,
        dumped=dumped,
    )

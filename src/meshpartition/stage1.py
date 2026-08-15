"""Stage 1 -- Component triage and piece apportionment.

Classify connected components (by shared-edge topology only -- no bridges
exist yet, those are Stage 2) as major (A_k >= beta * Abar) or satellite,
then apportion N pieces across the major components.

Satellite adoption isn't formally resolved until Stage 2 builds bridges, so
the "area of any satellites they will adopt" the spec calls for is
approximated here with a nearest-centroid heuristic. Stage 2 supersedes this
with the real anchor once bridges exist; apportionment can be re-run then if
the anchors disagree with this provisional host.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import WorkingMesh
from .util import UnionFind

DEFAULT_BETA = 0.25


@dataclass
class TriageResult:
    face_component_id: np.ndarray  # (F,) int64, dense component id per face
    component_areas: list[float]  # area per component, indexed by component id
    component_centroids: list[np.ndarray]  # area-weighted centroid per component
    major_ids: list[int]
    satellite_ids: list[int]
    satellite_host: dict[int, int]  # satellite component id -> provisional host major id
    piece_counts: dict[int, int]  # major component id -> apportioned piece count
    abar: float  # target area per piece (A_total / N)
    beta: float


def compute_face_components(working: WorkingMesh) -> tuple[np.ndarray, int]:
    """Connected components of the working mesh under shared-edge adjacency only."""
    uf = UnionFind(working.face_count)
    for faces in working.edge_faces.values():
        if len(faces) >= 2:
            first = faces[0]
            for f in faces[1:]:
                uf.union(first, f)

    roots = np.array([uf.find(i) for i in range(working.face_count)], dtype=np.int64)
    unique_roots = np.unique(roots)
    root_to_id = {root: idx for idx, root in enumerate(unique_roots)}
    face_component_id = np.array([root_to_id[r] for r in roots], dtype=np.int64)
    return face_component_id, len(unique_roots)


def _component_areas(
    working: WorkingMesh, face_component_id: np.ndarray, n_components: int
) -> list[float]:
    areas = [0.0] * n_components
    for face_id, comp_id in enumerate(face_component_id):
        areas[comp_id] += float(working.face_areas[face_id])
    return areas


def _component_centroid(
    working: WorkingMesh, face_component_id: np.ndarray, comp_id: int
) -> np.ndarray:
    mask = face_component_id == comp_id
    weights = working.face_areas[mask]
    points = working.face_centroids[mask]
    if weights.sum() <= 0.0:
        # All faces in this component are degenerate (zero area) -- fall back
        # to a plain average since area weighting is undefined.
        return points.mean(axis=0)
    return np.average(points, axis=0, weights=weights)


def _apportion(major_ids: list[int], weights: dict[int, float], n: int) -> dict[int, int]:
    """Largest-remainder apportionment: every major gets >=1 piece, the rest
    are handed out proportionally to weight, ties broken by ascending component id.
    """
    k = len(major_ids)
    counts = {c: 1 for c in major_ids}
    remaining = n - k
    if remaining <= 0:
        return counts

    total_weight = sum(weights[c] for c in major_ids) or 1.0
    quotas = {c: remaining * weights[c] / total_weight for c in major_ids}
    for c in major_ids:
        counts[c] += int(np.floor(quotas[c]))

    leftover = n - sum(counts.values())
    if leftover > 0:
        fractions = sorted(
            major_ids, key=lambda c: (-(quotas[c] - np.floor(quotas[c])), c)
        )
        for c in fractions[:leftover]:
            counts[c] += 1
    return counts


def triage(working: WorkingMesh, n_pieces: int, beta: float = DEFAULT_BETA) -> TriageResult:
    if n_pieces < 1:
        raise ValueError(f"n_pieces must be >= 1, got {n_pieces}")

    face_component_id, n_components = compute_face_components(working)
    component_areas = _component_areas(working, face_component_id, n_components)
    component_centroids = [
        _component_centroid(working, face_component_id, c) for c in range(n_components)
    ]

    abar = working.total_area() / n_pieces
    threshold = beta * abar
    major_ids = [c for c in range(n_components) if component_areas[c] >= threshold]
    satellite_ids = [c for c in range(n_components) if component_areas[c] < threshold]

    if not major_ids:
        raise ValueError(
            "No major component found: every component's area is below "
            f"beta * Abar ({threshold:.6g}). Lower beta or increase N."
        )
    if n_pieces < len(major_ids):
        raise ValueError(
            f"N={n_pieces} is too low: there are {len(major_ids)} major components, "
            "so at least that many pieces are required."
        )

    satellite_host: dict[int, int] = {}
    adopted_area: dict[int, float] = {c: 0.0 for c in major_ids}
    for s in satellite_ids:
        s_centroid = component_centroids[s]
        host = min(
            major_ids, key=lambda c: np.linalg.norm(component_centroids[c] - s_centroid)
        )
        satellite_host[s] = host
        adopted_area[host] += component_areas[s]

    weights = {c: component_areas[c] + adopted_area[c] for c in major_ids}
    piece_counts = _apportion(major_ids, weights, n_pieces)

    return TriageResult(
        face_component_id=face_component_id,
        component_areas=component_areas,
        component_centroids=component_centroids,
        major_ids=major_ids,
        satellite_ids=satellite_ids,
        satellite_host=satellite_host,
        piece_counts=piece_counts,
        abar=abar,
        beta=beta,
    )

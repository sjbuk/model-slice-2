"""Stage 0 -- Ingest and conditioning.

    1. Load geometry, UVs, normals, materials.               (mesh_io.load_obj)
    2. Remove faces with area below epsilon_area.
    3. Weld positions only, within delta_weld. Attributes stay per-corner.
    4. Build half-edge-equivalent adjacency, boundary edges, face areas/centroids.

GI-1 (area conservation) and GI-5 (determinism) are the invariants this
stage is responsible for establishing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .mesh import RawMesh, WorkingMesh
from .util import UnionFind, bbox_diagonal, triangle_areas, triangle_centroids

DEFAULT_WELD_FRACTION = 1e-4  # x bbox diagonal
DEFAULT_AREA_EPS_FRACTION = 1e-8  # x bbox diagonal squared


@dataclass
class IngestStats:
    original_face_count: int
    degenerate_face_count: int
    surviving_face_count: int
    original_vertex_count: int
    welded_vertex_count: int
    area_before: float  # sum of surviving (non-degenerate) face areas, pre-weld
    area_after: float  # sum of face areas, post-weld
    delta_weld: float
    epsilon_area: float


def default_tolerances(positions: np.ndarray) -> tuple[float, float]:
    diag = bbox_diagonal(positions)
    return DEFAULT_WELD_FRACTION * diag, DEFAULT_AREA_EPS_FRACTION * diag * diag


def _purge_degenerate_faces(raw: RawMesh, epsilon_area: float) -> tuple[RawMesh, np.ndarray, int]:
    areas = triangle_areas(raw.face_corner_positions())
    keep_mask = areas >= epsilon_area
    kept_original_indices = np.nonzero(keep_mask)[0]

    purged = RawMesh(
        positions=raw.positions,
        faces_pos=raw.faces_pos[keep_mask],
        uvs=raw.uvs,
        faces_uv=None if raw.faces_uv is None else raw.faces_uv[keep_mask],
        normals=raw.normals,
        faces_normal=None if raw.faces_normal is None else raw.faces_normal[keep_mask],
        material_ids=raw.material_ids[keep_mask],
    )
    degenerate_count = int(len(raw.faces_pos) - len(purged.faces_pos))
    return purged, kept_original_indices, degenerate_count


def _weld_positions(
    raw: RawMesh, delta_weld: float
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Merge positions within delta_weld. Does not touch UV/normal arrays.

    Returns (welded_positions, old_to_new_index, vertex_source_groups).
    """
    positions = raw.positions
    n = len(positions)
    uf = UnionFind(n)

    if n > 1 and delta_weld > 0:
        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=delta_weld, output_type="ndarray")
        # Sort for a deterministic union order (result is order-independent
        # anyway, but this keeps intermediate state reproducible).
        if len(pairs) > 0:
            order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            for i, j in pairs[order]:
                uf.union(int(i), int(j))

    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    unique_roots = np.unique(roots)
    # Dense id order follows ascending root (== ascending min original index),
    # which is stable given fixed input (A9 / GI-5).
    root_to_new = {root: new_id for new_id, root in enumerate(unique_roots)}
    old_to_new = np.array([root_to_new[r] for r in roots], dtype=np.int64)

    groups: list[list[int]] = [[] for _ in unique_roots]
    for old_idx in range(n):
        groups[old_to_new[old_idx]].append(old_idx)

    welded_positions = np.empty((len(unique_roots), 3), dtype=np.float64)
    for new_id, members in enumerate(groups):
        welded_positions[new_id] = positions[members].mean(axis=0)

    return welded_positions, old_to_new, groups


def _build_adjacency(
    faces_pos: np.ndarray,
) -> tuple[dict[tuple[int, int], list[int]], frozenset, frozenset]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, (a, b, c) in enumerate(faces_pos):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(u), int(v)) if u < v else (int(v), int(u))
            edge_faces.setdefault(key, []).append(face_id)

    boundary = frozenset(k for k, faces in edge_faces.items() if len(faces) == 1)
    nonmanifold = frozenset(k for k, faces in edge_faces.items() if len(faces) >= 3)
    return edge_faces, boundary, nonmanifold


def ingest(
    raw: RawMesh,
    delta_weld: float | None = None,
    epsilon_area: float | None = None,
) -> tuple[WorkingMesh, IngestStats]:
    default_delta, default_eps = default_tolerances(raw.positions)
    if delta_weld is None:
        delta_weld = default_delta
    if epsilon_area is None:
        epsilon_area = default_eps

    purged, kept_original_face_indices, degenerate_count = _purge_degenerate_faces(
        raw, epsilon_area
    )
    area_before = float(np.sum(triangle_areas(purged.face_corner_positions())))

    welded_positions, old_to_new, groups = _weld_positions(purged, delta_weld)
    welded_faces_pos = old_to_new[purged.faces_pos]

    corner_positions = welded_positions[welded_faces_pos]
    face_areas = triangle_areas(corner_positions)
    face_centroids = triangle_centroids(corner_positions)
    area_after = float(np.sum(face_areas))

    edge_faces, boundary_edges, nonmanifold_edges = _build_adjacency(welded_faces_pos)

    working = WorkingMesh(
        positions=welded_positions,
        faces_pos=welded_faces_pos,
        uvs=purged.uvs,
        faces_uv=purged.faces_uv,
        normals=purged.normals,
        faces_normal=purged.faces_normal,
        material_ids=purged.material_ids,
        face_areas=face_areas,
        face_centroids=face_centroids,
        edge_faces=edge_faces,
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        vertex_source_groups=groups,
        source_face_indices=kept_original_face_indices,
    )

    stats = IngestStats(
        original_face_count=raw.face_count,
        degenerate_face_count=degenerate_count,
        surviving_face_count=working.face_count,
        original_vertex_count=len(raw.positions),
        welded_vertex_count=working.vertex_count,
        area_before=area_before,
        area_after=area_after,
        delta_weld=delta_weld,
        epsilon_area=epsilon_area,
    )
    return working, stats

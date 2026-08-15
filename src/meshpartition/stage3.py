"""Stage 3 -- Refinement (Optimized).

Purpose: no single triangle should be so large it single-handedly blows the
25% area variance budget. Any face over a_max (default 0.25 x Abar) gets
longest-edge bisected.

Whichever face(s) currently share the exact position-edge being bisected are
all split at the same shared midpoint, together -- not just the offending
face alone. That is what keeps the mesh conforming (no T-junctions ever
introduced), regardless of whether the edge was manifold, boundary, or
already non-manifold going in.

Position midpoints are the exact geometric midpoint (no Laplacian smoothing),
so total area is conserved exactly (GI-1) -- each bisection splits a triangle
into two triangles of exactly half its area, sharing its apex and height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import WorkingMesh
from .stage0 import _build_adjacency
from .util import triangle_areas, triangle_centroids

DEFAULT_A_MAX_FRACTION = 0.25


@dataclass
class _FaceRecord:
    pos: tuple[int, int, int]
    uv: tuple[int, int, int] | None
    normal: tuple[int, int, int] | None
    material: int
    source_index: int


@dataclass
class RefineStats:
    a_max: float
    faces_split: int
    vertices_added: int
    passes: int


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _face_area(positions: list[np.ndarray], pos: tuple[int, int, int]) -> float:
    v0, v1, v2 = positions[pos[0]], positions[pos[1]], positions[pos[2]]
    return 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))


def _corner_order_for_edge(
    pos: tuple[int, int, int], edge: tuple[int, int]
) -> tuple[int, int, int]:
    """Find (i, j, k): i->j is a forward edge of the triangle matching `edge`, k is opposite."""
    for idx in range(3):
        a, b = pos[idx], pos[(idx + 1) % 3]
        if _edge_key(a, b) == edge:
            return idx, (idx + 1) % 3, (idx + 2) % 3
    raise ValueError(f"edge {edge} not found in face {pos}")


def _split_face(
    face: _FaceRecord,
    edge: tuple[int, int],
    midpoint_pos: int,
    midpoint_uv: int | None,
    midpoint_normal: int | None,
) -> tuple[_FaceRecord, _FaceRecord]:
    i, j, k = _corner_order_for_edge(face.pos, edge)

    def pick(triplet, mid):
        if triplet is None:
            return None, None
        return (triplet[i], mid, triplet[k]), (mid, triplet[j], triplet[k])

    pos1, pos2 = pick(face.pos, midpoint_pos)
    uv1, uv2 = pick(face.uv, midpoint_uv)
    normal1, normal2 = pick(face.normal, midpoint_normal)

    child1 = _FaceRecord(pos1, uv1, normal1, face.material, face.source_index)
    child2 = _FaceRecord(pos2, uv2, normal2, face.material, face.source_index)
    return child1, child2


def refine(
    working: WorkingMesh, abar: float, a_max_fraction: float = DEFAULT_A_MAX_FRACTION
) -> tuple[WorkingMesh, RefineStats]:
    a_max = a_max_fraction * abar

    positions: list[np.ndarray] = list(working.positions)
    uvs: list[np.ndarray] | None = list(working.uvs) if working.uvs is not None else None
    normals: list[np.ndarray] | None = (
        list(working.normals) if working.normals is not None else None
    )

    faces: dict[int, _FaceRecord] = {}
    edge_to_faces: dict[tuple[int, int], set[int]] = {}
    next_face_id = 0

    def register(face: _FaceRecord) -> int:
        nonlocal next_face_id
        fid = next_face_id
        next_face_id += 1
        faces[fid] = face
        for idx in range(3):
            key = _edge_key(face.pos[idx], face.pos[(idx + 1) % 3])
            edge_to_faces.setdefault(key, set()).add(fid)
        return fid

    def unregister(fid: int) -> None:
        face = faces.pop(fid)
        for idx in range(3):
            key = _edge_key(face.pos[idx], face.pos[(idx + 1) % 3])
            edge_to_faces[key].discard(fid)

    for i in range(working.face_count):
        uv = tuple(int(x) for x in working.faces_uv[i]) if working.faces_uv is not None else None
        normal = (
            tuple(int(x) for x in working.faces_normal[i])
            if working.faces_normal is not None
            else None
        )
        register(
            _FaceRecord(
                pos=tuple(int(x) for x in working.faces_pos[i]),
                uv=uv,
                normal=normal,
                material=int(working.material_ids[i]),
                source_index=int(working.source_face_indices[i]),
            )
        )

    edge_midpoint_cache: dict[tuple[int, int], int] = {}
    pending = {fid for fid, f in faces.items() if _face_area(positions, f.pos) > a_max}

    faces_split = 0
    passes = 0
    initial_vertex_count = len(positions)

    while pending:
        passes += 1
        fid = min(pending)
        pending.discard(fid)
        if fid not in faces:
            continue
        face = faces[fid]
        if _face_area(positions, face.pos) <= a_max:
            continue

        lengths = [
            (
                np.linalg.norm(positions[face.pos[idx]] - positions[face.pos[(idx + 1) % 3]]),
                _edge_key(face.pos[idx], face.pos[(idx + 1) % 3]),
            )
            for idx in range(3)
        ]
        longest_edge = max(lengths, key=lambda item: (item[0], item[1]))[1]

        if longest_edge not in edge_midpoint_cache:
            a, b = longest_edge
            new_pos = (positions[a] + positions[b]) / 2.0
            positions.append(new_pos)
            edge_midpoint_cache[longest_edge] = len(positions) - 1
        midpoint_pos = edge_midpoint_cache[longest_edge]

        affected_face_ids = sorted(edge_to_faces.get(longest_edge, set()))
        for gid in affected_face_ids:
            g = faces[gid]

            midpoint_uv = None
            if uvs is not None and g.uv is not None:
                i, j, _ = _corner_order_for_edge(g.pos, longest_edge)
                new_uv = (uvs[g.uv[i]] + uvs[g.uv[j]]) / 2.0
                uvs.append(new_uv)
                midpoint_uv = len(uvs) - 1

            midpoint_normal = None
            if normals is not None and g.normal is not None:
                i, j, _ = _corner_order_for_edge(g.pos, longest_edge)
                new_normal = (normals[g.normal[i]] + normals[g.normal[j]]) / 2.0
                positions_norm = np.linalg.norm(new_normal)
                if positions_norm > 1e-300:
                    new_normal = new_normal / positions_norm
                normals.append(new_normal)
                midpoint_normal = len(normals) - 1

            child1, child2 = _split_face(g, longest_edge, midpoint_pos, midpoint_uv, midpoint_normal)
            unregister(gid)
            faces_split += 1
            for child in (child1, child2):
                new_id = register(child)
                if _face_area(positions, child.pos) > a_max:
                    pending.add(new_id)

    final_ids = sorted(faces.keys())
    faces_pos = np.array([faces[i].pos for i in final_ids], dtype=np.int64)
    faces_uv = (
        np.array([faces[i].uv for i in final_ids], dtype=np.int64) if uvs is not None else None
    )
    faces_normal = (
        np.array([faces[i].normal for i in final_ids], dtype=np.int64)
        if normals is not None
        else None
    )
    material_ids = np.array([faces[i].material for i in final_ids], dtype=np.int64)
    source_face_indices = np.array([faces[i].source_index for i in final_ids], dtype=np.int64)

    final_positions = np.array(positions, dtype=np.float64)
    final_uvs = np.array(uvs, dtype=np.float64) if uvs is not None else None
    final_normals = np.array(normals, dtype=np.float64) if normals is not None else None

    edge_faces, boundary_edges, nonmanifold_edges = _build_adjacency(faces_pos)
    corner_positions = final_positions[faces_pos]
    face_areas = triangle_areas(corner_positions)
    face_centroids = triangle_centroids(corner_positions)

    vertex_source_groups = list(working.vertex_source_groups) + [
        [] for _ in range(len(final_positions) - initial_vertex_count)
    ]

    refined = WorkingMesh(
        positions=final_positions,
        faces_pos=faces_pos,
        uvs=final_uvs,
        faces_uv=faces_uv,
        normals=final_normals,
        faces_normal=faces_normal,
        material_ids=material_ids,
        face_areas=face_areas,
        face_centroids=face_centroids,
        edge_faces=edge_faces,
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        vertex_source_groups=vertex_source_groups,
        source_face_indices=source_face_indices,
    )

    stats = RefineStats(
        a_max=a_max,
        faces_split=faces_split,
        vertices_added=len(final_positions) - initial_vertex_count,
        passes=passes,
    )
    return refined, stats

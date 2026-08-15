"""Stage 2 -- Bridge construction.

    1. Build a BVH over all faces.                     (brute force at this scale; see note below)
    2. For each satellite, find the m nearest distinct points on other components.
    3. Reject hits beyond distance tau_S or where normal agreement is below theta_n.
    4. Enclosure override via generalized winding number, so nested objects
       (gems) bridge to their settings, not merely the nearest surface.
    5. Emit surviving hits as bridge edges; the best is the satellite's anchor.

A "BVH" is a performance device, not a semantic one -- it changes how fast you
find the nearest triangle, never which triangle is nearest. At the fixture
scale here (tens to hundreds of faces), brute-force closest-point-on-triangle
over all candidate faces gives identical results to a tree-accelerated search,
so that's what's implemented; swap in an actual spatial index (nanoflann/BVH)
before scaling to production meshes.

Bridges are only ever built from a satellite to a MAJOR component -- never
satellite-to-satellite -- so every anchor by construction terminates on a
major component (spec's "anchor termination" requirement).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import WorkingMesh
from .stage1 import TriageResult
from .util import (
    bbox_diagonal,
    closest_points_on_triangles,
    face_normals,
    generalized_winding_number,
)

DEFAULT_C_TAU = 0.5
DEFAULT_TAU_MAX_FRACTION = 0.03
DEFAULT_M = 5
DEFAULT_THETA_N = 0.0
DEFAULT_ENCLOSURE_THRESHOLD = 0.5


@dataclass
class BridgeEdge:
    satellite_component: int
    satellite_face: int
    other_component: int
    other_face: int
    distance: float
    normal_agreement: float
    via_enclosure: bool = False


@dataclass
class BridgeResult:
    bridges: dict[int, list[BridgeEdge]]  # satellite component id -> surviving edges
    anchors: dict[int, BridgeEdge]  # satellite component id -> chosen anchor
    promoted_ids: list[int]  # satellites with no valid bridge, now treated as major
    major_ids: list[int]  # triage.major_ids + promoted_ids
    satellite_ids: list[int]  # triage.satellite_ids - promoted_ids
    winding_numbers: dict[tuple[int, int], float] = field(default_factory=dict)


def _satellite_gap_threshold(
    working: WorkingMesh,
    face_component_id: np.ndarray,
    satellite_id: int,
    c_tau: float,
    tau_max_fraction: float,
    global_bbox_diag: float,
) -> float:
    mask = face_component_id == satellite_id
    sat_positions = working.positions[np.unique(working.faces_pos[mask])]
    local_scale = bbox_diagonal(sat_positions)
    return min(c_tau * local_scale, tau_max_fraction * global_bbox_diag)


def _nearest_distinct_hits(
    working: WorkingMesh,
    normals: np.ndarray,
    satellite_faces: np.ndarray,
    other_faces: np.ndarray,
    m: int,
) -> list[tuple[float, int, int, float]]:
    """Returns up to m (distance, satellite_face, other_face, normal_agreement)
    tuples, sorted ascending by distance, one entry per distinct other_face.
    """
    if len(satellite_faces) == 0 or len(other_faces) == 0:
        return []

    other_tri = working.face_corner_positions()[other_faces]
    a, b, c = other_tri[:, 0], other_tri[:, 1], other_tri[:, 2]

    best_per_other_face: dict[int, tuple[float, int, float]] = {}
    for sat_face in satellite_faces:
        query = working.face_centroids[sat_face]
        query_batch = np.broadcast_to(query, a.shape)
        _, dists = closest_points_on_triangles(query_batch, a, b, c)
        for local_idx, other_face in enumerate(other_faces):
            dist = float(dists[local_idx])
            existing = best_per_other_face.get(other_face)
            if existing is None or dist < existing[0]:
                agreement = float(np.dot(normals[sat_face], normals[other_face]))
                best_per_other_face[other_face] = (dist, sat_face, agreement)

    hits = [
        (dist, sat_face, other_face, agreement)
        for other_face, (dist, sat_face, agreement) in best_per_other_face.items()
    ]
    hits.sort(key=lambda h: (h[0], h[2]))
    return hits[:m]


def build_bridges(
    working: WorkingMesh,
    triage: TriageResult,
    m: int = DEFAULT_M,
    c_tau: float = DEFAULT_C_TAU,
    tau_max_fraction: float = DEFAULT_TAU_MAX_FRACTION,
    theta_n: float = DEFAULT_THETA_N,
    enclosure_threshold: float = DEFAULT_ENCLOSURE_THRESHOLD,
) -> BridgeResult:
    normals = face_normals(working.face_corner_positions())
    global_bbox_diag = bbox_diagonal(working.positions)
    face_component_id = triage.face_component_id

    major_face_masks = {
        major: np.nonzero(face_component_id == major)[0] for major in triage.major_ids
    }

    bridges: dict[int, list[BridgeEdge]] = {}
    anchors: dict[int, BridgeEdge] = {}
    promoted_ids: list[int] = []
    winding_numbers: dict[tuple[int, int], float] = {}

    for satellite in triage.satellite_ids:
        satellite_faces = np.nonzero(face_component_id == satellite)[0]
        other_faces = np.concatenate(
            [major_face_masks[major] for major in triage.major_ids if len(major_face_masks[major])]
        )

        hits = _nearest_distinct_hits(working, normals, satellite_faces, other_faces, m)

        tau_s = _satellite_gap_threshold(
            working, face_component_id, satellite, c_tau, tau_max_fraction, global_bbox_diag
        )
        survivors = [
            BridgeEdge(
                satellite_component=satellite,
                satellite_face=sat_face,
                other_component=int(face_component_id[other_face]),
                other_face=other_face,
                distance=dist,
                normal_agreement=agreement,
            )
            for dist, sat_face, other_face, agreement in hits
            if dist <= tau_s and agreement >= theta_n
        ]

        satellite_centroid = triage.component_centroids[satellite]
        enclosing_major = None
        best_wn = -1.0
        for major in triage.major_ids:
            wn = generalized_winding_number(
                satellite_centroid, working.positions, working.faces_pos[major_face_masks[major]]
            )
            winding_numbers[(satellite, major)] = wn
            if wn > enclosure_threshold and wn > best_wn:
                enclosing_major = major
                best_wn = wn

        if enclosing_major is not None:
            enclosing_faces = major_face_masks[enclosing_major]
            enclosure_hits = _nearest_distinct_hits(
                working, normals, satellite_faces, enclosing_faces, m=1
            )
            dist, sat_face, other_face, agreement = enclosure_hits[0]
            anchor = BridgeEdge(
                satellite_component=satellite,
                satellite_face=sat_face,
                other_component=enclosing_major,
                other_face=other_face,
                distance=dist,
                normal_agreement=agreement,
                via_enclosure=True,
            )
            if not any(edge.other_component == enclosing_major for edge in survivors):
                survivors.append(anchor)
            bridges[satellite] = survivors
            anchors[satellite] = anchor
            continue

        if survivors:
            bridges[satellite] = survivors
            anchors[satellite] = min(survivors, key=lambda e: e.distance)
        else:
            promoted_ids.append(satellite)

    updated_major_ids = sorted(triage.major_ids + promoted_ids)
    updated_satellite_ids = [s for s in triage.satellite_ids if s not in promoted_ids]

    return BridgeResult(
        bridges=bridges,
        anchors=anchors,
        promoted_ids=promoted_ids,
        major_ids=updated_major_ids,
        satellite_ids=updated_satellite_ids,
        winding_numbers=winding_numbers,
    )

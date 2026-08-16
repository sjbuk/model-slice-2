"""Stage 2 -- Bridge construction.

    1. Build a BVH over all faces.                     (KDTree candidate prefilter; see note below)
    2. For each satellite, find the m nearest distinct points on other components.
    3. Reject hits beyond distance tau_S or where normal agreement is below theta_n.
    4. Enclosure override via generalized winding number, so nested objects
       (gems) bridge to their settings, not merely the nearest surface.
    5. Emit surviving hits as bridge edges; the best is the satellite's anchor.

A "BVH" is a performance device, not a semantic one -- it changes how fast you
find the nearest triangle, never which triangle is nearest. Exact
closest-point-on-triangle is still brute force (region-based, see
util.closest_point_distance_matrix), but which major faces are even
considered for a given satellite is now bounded by a KDTree query over major
face centroids, radius = tau_s (the satellite's own gap threshold) plus a
margin covering both faces' extents -- any major face that could possibly
land within tau_s of some satellite face is guaranteed to be in the candidate
set, so results are identical to unbounded brute force, just without
evaluating faces that are provably too far away to ever survive the gap
check. This matters once components number in the hundreds and majors span
tens of thousands of faces (e.g. a real hard-surface asset): unbounded brute
force is O(sum of satellite faces x total major faces), which is billions of
triangle-distance evaluations and effectively never finishes. The exact
distance evaluations that do survive the prefilter run through a
numba-compiled kernel rather than vectorized numpy, for the same reason
(numpy's vectorized region-based test allocates ~20 temporary arrays per
call; the compiled version allocates none).

Bridges are only ever built from a satellite to a MAJOR component -- never
satellite-to-satellite -- so every anchor by construction terminates on a
major component (spec's "anchor termination" requirement).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .mesh import WorkingMesh
from .stage1 import TriageResult
from .util import (
    bbox_diagonal,
    closest_point_distance_matrix,
    face_normals,
    generalized_winding_number,
)

# c_tau=0.5 rejects real, flush-fitting detail parts (lens layers, rivets,
# trim discs -- common in hard-surface assets that were never vertex-welded
# across material boundaries): a small satellite's own bbox diagonal can be
# on the same order as its unwelded seam gap, so a gap allowance of half its
# own size is tighter than the model's overall scale would justify. 1.25
# still lets the global cap (tau_max_fraction) be the real backstop against
# satellites that are actually far away, in local-size terms as well as
# global (Stage 2.1's spec case: a button moved 10m off a ~1m-diagonal shirt
# stays promoted at any c_tau in this range, since 10m dwarfs both terms).
DEFAULT_C_TAU = 1.25
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
    forced: bool = False  # force_exact_count fallback: nearest-centroid host, no gap/normal check


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


def _face_extents(corner_positions: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(F,) max distance from each face's centroid to any of its own corners.

    Used to grow a centroid-to-centroid spatial query radius by enough that
    it can't miss a face whose actual (corner-to-corner) closest point is
    within range even though its centroid looks too far away.
    """
    return np.max(np.linalg.norm(corner_positions - centroids[:, None, :], axis=2), axis=1)


# closest_point_distance_matrix is compiled and allocates nothing beyond its
# (s, t) output, so chunking here exists purely to bound that one array's
# memory for pathological satellite/candidate counts, not to amortize
# per-call overhead the way it did for the old vectorized-numpy version.
_CHUNK_ELEMENT_BUDGET = 50_000_000


def _nearest_distinct_hits(
    corner_positions: np.ndarray,
    centroids: np.ndarray,
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

    other_tri = corner_positions[other_faces]
    a0, b0, c0 = other_tri[:, 0], other_tri[:, 1], other_tri[:, 2]
    other_normals = normals[other_faces]

    t = len(other_faces)
    best_dist = np.full(t, np.inf)
    best_sat_face = np.full(t, -1, dtype=np.int64)
    best_agreement = np.zeros(t)

    chunk_size = max(1, _CHUNK_ELEMENT_BUDGET // t)

    # Which of the batched satellite faces wins for each of the t candidates
    # is a plain columnwise min -- tracked with vectorized compare-and-update
    # instead of a Python dict keyed by other_face (S*T individual dict
    # lookups, the original bottleneck here).
    for start in range(0, len(satellite_faces), chunk_size):
        chunk = satellite_faces[start : start + chunk_size]
        s = len(chunk)

        dists = closest_point_distance_matrix(centroids[chunk], a0, b0, c0)

        local_best_row = np.argmin(dists, axis=0)
        cols = np.arange(t)
        local_best_dist = dists[local_best_row, cols]
        mask = local_best_dist < best_dist
        if np.any(mask):
            local_best_sat_face = chunk[local_best_row]
            agreement = normals[chunk] @ other_normals.T
            local_best_agreement = agreement[local_best_row, cols]
            best_dist = np.where(mask, local_best_dist, best_dist)
            best_sat_face = np.where(mask, local_best_sat_face, best_sat_face)
            best_agreement = np.where(mask, local_best_agreement, best_agreement)

    hits = [
        (float(best_dist[i]), int(best_sat_face[i]), int(other_faces[i]), float(best_agreement[i]))
        for i in range(t)
        if best_sat_face[i] >= 0
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
    force_exact_count: bool = False,
) -> BridgeResult:
    """force_exact_count=True disables promotion: a satellite that fails the
    gap/normal-agreement test force-adopts Stage 1's provisional
    nearest-centroid host instead of becoming its own major. This trades
    bridge plausibility for an exact N -- appropriate when pieces never need
    to read as physically/visually coherent on their own (e.g. VR puzzle
    pieces are just grouped geometry, not objects that must look sensibly
    assembled from any single piece's silhouette). Default behavior
    (promotion) is unchanged; this is opt-in per call.
    """
    corner_positions = working.face_corner_positions()
    centroids = working.face_centroids
    normals = face_normals(corner_positions)
    global_bbox_diag = bbox_diagonal(working.positions)
    face_component_id = triage.face_component_id

    major_face_masks = {
        major: np.nonzero(face_component_id == major)[0] for major in triage.major_ids
    }

    all_major_faces = np.concatenate(
        [major_face_masks[major] for major in triage.major_ids if len(major_face_masks[major])]
    )
    face_extents = _face_extents(corner_positions, centroids)
    major_tree = cKDTree(centroids[all_major_faces])

    # A query margin has to cover the largest major face that could possibly
    # be a survivor, but sizing it off the true max lets a handful of
    # unusually large faces (a car body panel vs. hundreds of bolts) blow up
    # every query's radius, even for satellites nowhere near them. Size off
    # the 99th percentile instead, and give the rare larger-than-that faces
    # (~1% of majors) their own small KDTree with a margin sized to *their*
    # extent -- still a per-satellite proximity check, just against a tiny
    # tree, rather than splicing all of them into every satellite's
    # candidate set regardless of whether they're anywhere nearby.
    major_extents = face_extents[all_major_faces]
    typical_major_extent = float(np.percentile(major_extents, 99))
    outlier_mask = major_extents > typical_major_extent
    outlier_major_faces = all_major_faces[outlier_mask]
    outlier_tree = cKDTree(centroids[outlier_major_faces]) if len(outlier_major_faces) else None
    outlier_max_extent = float(major_extents[outlier_mask].max()) if len(outlier_major_faces) else 0.0

    bridges: dict[int, list[BridgeEdge]] = {}
    anchors: dict[int, BridgeEdge] = {}
    promoted_ids: list[int] = []
    winding_numbers: dict[tuple[int, int], float] = {}

    for satellite in triage.satellite_ids:
        satellite_faces = np.nonzero(face_component_id == satellite)[0]

        tau_s = _satellite_gap_threshold(
            working, face_component_id, satellite, c_tau, tau_max_fraction, global_bbox_diag
        )

        # Per satellite *face*, not one bounding sphere for the whole
        # satellite: some satellites are spread across widely separated
        # points on the source asset (e.g. a decal set with pieces on both
        # sides of a car), so a single bounding sphere covering the whole
        # component's extent would swallow up the entire model in between.
        # A major face that could possibly land within tau_s of a given
        # satellite face has its centroid within
        # tau_s + (both faces' own centroid-to-corner extent) of that
        # satellite face's centroid, so a query at that radius per face can't
        # miss a true survivor. Extra candidates beyond the true gap are
        # harmless: the exact tau_s check below still filters them out.
        sat_centroids = centroids[satellite_faces]
        sat_extents = face_extents[satellite_faces]

        radii = tau_s + sat_extents + typical_major_extent
        neighbor_lists = major_tree.query_ball_point(sat_centroids, radii)
        candidate_local = {idx for neighbors in neighbor_lists for idx in neighbors}
        other_faces = (
            all_major_faces[sorted(candidate_local)]
            if candidate_local
            else np.empty(0, dtype=all_major_faces.dtype)
        )

        if outlier_tree is not None:
            outlier_radii = tau_s + sat_extents + outlier_max_extent
            outlier_neighbor_lists = outlier_tree.query_ball_point(sat_centroids, outlier_radii)
            outlier_local = {idx for neighbors in outlier_neighbor_lists for idx in neighbors}
            if outlier_local:
                other_faces = np.union1d(other_faces, outlier_major_faces[sorted(outlier_local)])

        hits = _nearest_distinct_hits(corner_positions, centroids, normals, satellite_faces, other_faces, m)

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
                corner_positions, centroids, normals, satellite_faces, enclosing_faces, m=1
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
            continue

        if force_exact_count:
            host = triage.satellite_host[satellite]
            host_faces = major_face_masks[host]
            forced_hits = _nearest_distinct_hits(
                corner_positions, centroids, normals, satellite_faces, host_faces, m=1
            )
            dist, sat_face, other_face, agreement = forced_hits[0]
            anchor = BridgeEdge(
                satellite_component=satellite,
                satellite_face=sat_face,
                other_component=host,
                other_face=other_face,
                distance=dist,
                normal_agreement=agreement,
                forced=True,
            )
            bridges[satellite] = [anchor]
            anchors[satellite] = anchor
            continue

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

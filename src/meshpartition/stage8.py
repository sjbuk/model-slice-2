"""Stage 8 -- Lloyd relaxation.

    1. Re-seed at region medoids, re-run Stage 6, re-run Stage 7.
    2. Score the iteration based on area deviation, seam length, and elongation.
    3. Stop when max(|A_i - Abar|)/Abar <= epsilon_bal (25%), or I_max is
       reached. Retain the best-scoring iteration.

A region's "medoid" is the face minimizing the sum of dual-graph distances to
every other face currently carrying that region's label -- the graph analogue
of a geometric medoid, computed exactly (all-pairs within the region) since
fixture-scale regions are small. This is done over the region's *current*
label, not its originating major component: after Stage 7 a region's
footprint can include adopted satellite cells and dumped-in cells from
elsewhere, all reachable only through the full dual graph (surface + bridge
edges).

Re-seeding keeps the same region id -> its new medoid face, preserving
region identity across iterations (needed for both the score history and for
Stage 7's next_alpha to mean anything: it is carried forward per region, not
reset each pass).

Scoring: the spec asks for area deviation, seam length, and elongation but
does not prescribe a formula. This uses an unweighted sum of three
non-negative terms, each 0 for an ideal iteration: max_i |A_i - Abar| / Abar
(area), total cross-label edge length / bbox diagonal (seam -- grows with
mesh resolution and region count, so it's only meaningful for comparing
iterations of the same run, not across different N or fixtures), and
mean_i(rho_i) - 1 (elongation, capped per-region before averaging -- see
ELONGATION_SCORE_CAP -- so one degenerate sliver region can't swamp the
whole score). Lower is better.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .mesh import WorkingMesh
from .stage1 import TriageResult
from .stage2 import BridgeResult
from .stage4 import DualGraph
from .stage5 import SeedResult, _multi_source_distances
from .stage6 import GrowthResult, grow
from .stage7 import DEFAULT_GAMMA, RepairResult, repair
from .util import bbox_diagonal

DEFAULT_EPSILON_BAL = 0.25
DEFAULT_I_MAX = 15


@dataclass
class IterationRecord:
    label: np.ndarray
    region_area: np.ndarray
    elongation: np.ndarray
    next_alpha: np.ndarray
    max_area_deviation: float
    seam_length: float
    score: float


@dataclass
class RelaxationResult:
    label: np.ndarray
    region_area: np.ndarray
    max_area_deviation: float
    score: float
    iterations_run: int
    converged: bool
    scores: list[float] = field(default_factory=list)  # per iteration, in order


def _max_area_deviation(region_area: np.ndarray, abar: float) -> float:
    return float(np.max(np.abs(region_area - abar)) / abar)


def _seam_length(working: WorkingMesh, label: np.ndarray) -> float:
    total = 0.0
    for (v0, v1), faces in working.edge_faces.items():
        if len({int(label[f]) for f in faces}) > 1:
            total += float(np.linalg.norm(working.positions[v1] - working.positions[v0]))
    return total


ELONGATION_SCORE_CAP = 20.0  # keeps one degenerate sliver region from swamping the whole score


def _score(max_area_deviation: float, seam_length: float, elongation: np.ndarray, bbox_diag: float) -> float:
    seam_term = seam_length / bbox_diag if bbox_diag > 0 else 0.0
    # elongation is unbounded in principle -- e.g. any region reduced to exactly
    # two faces is a mathematically perfect line (rank-1 centroid covariance),
    # an arbitrarily large ratio. Capping and normalizing to [0, 1] keeps that
    # single degenerate region from dominating the sum the way an uncapped
    # mean would.
    capped = np.minimum(elongation, ELONGATION_SCORE_CAP)
    elongation_term = float(np.mean(capped) - 1.0) / (ELONGATION_SCORE_CAP - 1.0)
    return max_area_deviation + seam_term + elongation_term


def _record(working: WorkingMesh, repaired: RepairResult, abar: float, bbox_diag: float) -> IterationRecord:
    max_dev = _max_area_deviation(repaired.region_area, abar)
    seam = _seam_length(working, repaired.label)
    score = _score(max_dev, seam, repaired.elongation, bbox_diag)
    return IterationRecord(
        label=repaired.label,
        region_area=repaired.region_area,
        elongation=repaired.elongation,
        next_alpha=repaired.next_alpha,
        max_area_deviation=max_dev,
        seam_length=seam,
        score=score,
    )


def _region_adjacency(dual_graph: DualGraph, label: np.ndarray, region_id: int) -> dict[int, list[tuple[int, float]]]:
    face_set = {int(f) for f in np.nonzero(label == region_id)[0]}
    adjacency: dict[int, list[tuple[int, float]]] = {f: [] for f in face_set}
    for f in face_set:
        for neighbor, cost, _is_bridge in dual_graph.adjacency[f]:
            if neighbor in face_set:
                adjacency[f].append((neighbor, cost))
    return adjacency


def _medoid(adjacency: dict[int, list[tuple[int, float]]]) -> int:
    faces = sorted(adjacency.keys())
    best_face = faces[0]
    best_total = math.inf
    for f in faces:
        total = sum(_multi_source_distances(adjacency, [f]).values())
        if total < best_total:
            best_total = total
            best_face = f
    return best_face


def _reseed(seed_result: SeedResult, new_seed_faces: list[int]) -> SeedResult:
    seeds: dict[int, list[int]] = {}
    seed_owner: dict[int, int] = {}
    idx = 0
    for major in sorted(seed_result.seeds):
        count = len(seed_result.seeds[major])
        chosen = new_seed_faces[idx : idx + count]
        idx += count
        seeds[major] = chosen
        for face_id in chosen:
            seed_owner[face_id] = major
    return SeedResult(piece_counts=seed_result.piece_counts, seeds=seeds, seed_owner=seed_owner)


def relax(
    working: WorkingMesh,
    dual_graph: DualGraph,
    triage: TriageResult,
    bridges: BridgeResult,
    seed_result: SeedResult,
    growth: GrowthResult,
    repaired: RepairResult,
    epsilon_bal: float = DEFAULT_EPSILON_BAL,
    i_max: int = DEFAULT_I_MAX,
    gamma: float = DEFAULT_GAMMA,
) -> RelaxationResult:
    abar = triage.abar
    bbox_diag = bbox_diagonal(working.positions)
    n_regions = len(growth.region_seed)

    records = [_record(working, repaired, abar, bbox_diag)]
    current_seed_result = seed_result
    current_alpha = repaired.next_alpha
    iterations_run = 1

    while records[-1].max_area_deviation > epsilon_bal and iterations_run < i_max:
        medoid_faces = [
            _medoid(_region_adjacency(dual_graph, records[-1].label, region_id))
            for region_id in range(n_regions)
        ]
        current_seed_result = _reseed(current_seed_result, medoid_faces)

        new_growth = grow(working, dual_graph, current_seed_result, abar=abar, alpha=current_alpha)
        new_repaired = repair(working, dual_graph, triage, bridges, new_growth, alpha=current_alpha, gamma=gamma)

        records.append(_record(working, new_repaired, abar, bbox_diag))
        current_alpha = new_repaired.next_alpha
        iterations_run += 1

    best = min(records, key=lambda r: r.score)

    return RelaxationResult(
        label=best.label,
        region_area=best.region_area,
        max_area_deviation=best.max_area_deviation,
        score=best.score,
        iterations_run=iterations_run,
        converged=records[-1].max_area_deviation <= epsilon_bal,
        scores=[r.score for r in records],
    )

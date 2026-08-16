"""Stage 8 -- Lloyd relaxation.

    1. Re-seed at region medoids, re-run Stage 6, re-run Stage 7.
    2. Score the iteration based on area deviation, seam length, and elongation.
    3. Stop when max(|A_i - Abar|)/Abar <= epsilon_bal (25%), or I_max is
       reached. Retain the best-scoring iteration.

A region's "medoid" is the face minimizing the sum of dual-graph distances to
every other face currently carrying that region's label -- the graph analogue
of a geometric medoid, computed exactly (all-pairs within the region) at
fixture scale; see _medoid's own docstring-comment for the approximation used
past _EXACT_MEDOID_MAX_FACES. This is done over the region's *current* label,
not its originating major component: after Stage 7 a region's footprint can
include adopted satellite cells and dumped-in cells from elsewhere, all
reachable only through the full dual graph (surface + bridge edges).

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

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .mesh import WorkingMesh
from .progress import ProgressReporter
from .stage1 import TriageResult
from .stage2 import BridgeResult
from .stage4 import DualGraph
from .stage5 import SeedResult
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
    iteration_labels: list[np.ndarray] = field(default_factory=list)  # per iteration, in order
    iteration_max_area_deviations: list[float] = field(default_factory=list)  # per iteration, in order
    best_iteration_index: int = 0  # index into the lists above of the retained (lowest-score) iteration


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


def _region_csr(adjacency: dict[int, list[tuple[int, float]]]) -> tuple[csr_matrix, list[int]]:
    faces = sorted(adjacency.keys())
    local_index = {f: i for i, f in enumerate(faces)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for f, neighbors in adjacency.items():
        for neighbor, cost in neighbors:
            rows.append(local_index[f])
            cols.append(local_index[neighbor])
            data.append(cost)
    n = len(faces)
    csr = csr_matrix((data, (rows, cols)), shape=(n, n))
    return csr, faces


# Exact all-pairs Dijkstra (scipy's dijkstra() with no `indices`) materializes
# a dense n x n distance matrix -- fine at "fixture-scale" (the original
# design target) but a real hard-surface asset can leave a region with tens
# of thousands of faces (e.g. force_exact_count on a mesh with hundreds of
# disconnected detail parts and few majors), where that matrix alone is
# gigabytes and the O(n * (E+n) log n) time to fill it is minutes, every
# relaxation iteration. Past this many faces, approximate instead.
_EXACT_MEDOID_MAX_FACES = 1500
# Each candidate is scored by one single-source Dijkstra (its true, exact sum
# of distances to the rest of the region -- O(n) memory, since it's one row,
# not the whole matrix), so this bounds the approximation to a fixed number
# of exact evaluations rather than evaluating every face as a candidate.
_APPROX_MEDOID_SAMPLE_SIZE = 200


def _medoid(adjacency: dict[int, list[tuple[int, float]]]) -> int:
    csr, faces = _region_csr(adjacency)
    n = len(faces)

    if n <= _EXACT_MEDOID_MAX_FACES:
        # adjacency already carries both directions of every edge (see build_dual_graph),
        # so directed=True over those explicit reciprocal entries gives correct
        # undirected distances without scipy needing to symmetrize the matrix itself.
        distances = dijkstra(csr, directed=True)
        totals = distances.sum(axis=1)
        return faces[int(np.argmin(totals))]

    # Deterministic, evenly-spaced sample of candidates rather than every
    # face -- Lloyd relaxation only needs a reasonable re-seed point each
    # pass, not the true minimum over all n, and the loop self-corrects
    # over iterations regardless.
    stride = max(1, n // _APPROX_MEDOID_SAMPLE_SIZE)
    best_face = faces[0]
    best_total = np.inf
    for idx in range(0, n, stride):
        total = float(dijkstra(csr, directed=True, indices=idx).sum())
        if total < best_total:
            best_total = total
            best_face = faces[idx]
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
    progress: ProgressReporter | None = None,
) -> RelaxationResult:
    abar = triage.abar
    bbox_diag = bbox_diagonal(working.positions)
    n_regions = len(growth.region_seed)

    records = [_record(working, repaired, abar, bbox_diag)]
    current_seed_result = seed_result
    current_alpha = repaired.next_alpha
    iterations_run = 1

    # The while loop runs at most (i_max - 1) more passes (iterations_run
    # starts at 1), each computing one medoid per region -- report substep
    # progress as a single counter over all of those medoid computations
    # combined, rather than restarting a 1..n_regions bar every pass. A
    # per-pass-only bar looks identical on iteration 1 and iteration 14,
    # giving no sense of overall relax progress; this counter instead climbs
    # monotonically across the whole phase, with the label carrying which
    # Lloyd iteration is currently in flight.
    total_medoid_steps = (i_max - 1) * n_regions

    while records[-1].max_area_deviation > epsilon_bal and iterations_run < i_max:
        pass_index = iterations_run - 1  # 0-based count of passes completed so far
        iteration_label = f"relax iteration {iterations_run + 1}/{i_max}"
        if progress is not None:
            progress.substep(pass_index * n_regions, total_medoid_steps, label=iteration_label)

        medoid_faces = []
        for region_id in range(n_regions):
            medoid_faces.append(_medoid(_region_adjacency(dual_graph, records[-1].label, region_id)))
            if progress is not None:
                progress.substep(
                    pass_index * n_regions + region_id + 1, total_medoid_steps, label=iteration_label
                )
        current_seed_result = _reseed(current_seed_result, medoid_faces)

        new_growth = grow(working, dual_graph, current_seed_result, abar=abar, alpha=current_alpha)
        new_repaired = repair(working, dual_graph, triage, bridges, new_growth, alpha=current_alpha, gamma=gamma)

        records.append(_record(working, new_repaired, abar, bbox_diag))
        current_alpha = new_repaired.next_alpha
        iterations_run += 1

    best_index, best = min(enumerate(records), key=lambda p: p[1].score)

    return RelaxationResult(
        label=best.label,
        region_area=best.region_area,
        max_area_deviation=best.max_area_deviation,
        score=best.score,
        iterations_run=iterations_run,
        converged=records[-1].max_area_deviation <= epsilon_bal,
        scores=[r.score for r in records],
        iteration_labels=[r.label for r in records],
        iteration_max_area_deviations=[r.max_area_deviation for r in records],
        best_iteration_index=best_index,
    )

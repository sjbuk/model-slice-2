"""Stage 6 -- Capacity-constrained growth.

Single global priority queue over every seed from Stage 5, competing for
faces across the whole dual graph (including bridge edges -- a region may
grow into satellite territory; Stage 7 overrides that for satellites
specifically via the anchor fixup, so it is harmless if it happens here).

The claim key for face f by region i is

    key = d_i(f) * (max(A_i, zeta * Abar) / Abar) ** alpha

where d_i(f) is i's own dual-graph shortest-path distance to f (each region
runs its own independent single-source Dijkstra, all sharing one heap so the
globally cheapest pending claim always goes next) and A_i is region i's
claimed area so far.

Lazy-evaluation fix: A_i only grows during the run, so a key computed at push
time is a lower bound that can go stale before the entry is popped. On pop,
the key is recomputed from the CURRENT A_i; if that is worse (larger) than
the stored key, the entry is re-pushed with the corrected key instead of
being allowed to claim the face, and the loop moves on to the next-cheapest
item. This is also why the popped claim-key sequence is non-decreasing over
the whole run: exactly the same monotonicity argument as Dijkstra's own
finalized-distance property, just carried out per contested face instead of
per single source.

A second, ordinary Dijkstra staleness check also applies: a region can reach
the same face via more than one path before either is popped, so only the
cheapest known (region, face) distance is kept live; a popped entry whose
distance no longer matches that record was already superseded and is simply
dropped (not re-pushed -- the better entry for that pair is already queued
or claimed).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .mesh import WorkingMesh
from .stage4 import DualGraph
from .stage5 import SeedResult

DEFAULT_ALPHA = 1.5
DEFAULT_ZETA = 0.1


@dataclass
class GrowthResult:
    label: np.ndarray  # (F,) int64, region id per face
    region_area: np.ndarray  # (R,) float64, final claimed area per region
    region_seed: list[int]  # region id -> seed face id
    claims: list[tuple[int, int, float]] = field(default_factory=list)  # (face_id, region_id, key), in claim order
    stale_requeues: int = 0  # number of times a popped entry's key was found stale and re-pushed


def _capacity_factor(area: float, abar: float, zeta: float, alpha: float) -> float:
    return (max(area, zeta * abar) / abar) ** alpha


def grow(
    working: WorkingMesh,
    dual_graph: DualGraph,
    seed_result: SeedResult,
    abar: float,
    alpha: float | np.ndarray = DEFAULT_ALPHA,
    zeta: float = DEFAULT_ZETA,
) -> GrowthResult:
    region_seed: list[int] = []
    for major in sorted(seed_result.seeds):
        region_seed.extend(seed_result.seeds[major])
    n_regions = len(region_seed)

    # alpha may be a single value shared by every region (Stage 6 on its own)
    # or one value per region (Stage 8 feeding back Stage 7's elongation-raised
    # next_alpha for each region individually).
    alpha_per_region = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (n_regions,))

    label = np.full(working.face_count, -1, dtype=np.int64)
    region_area = np.zeros(n_regions, dtype=np.float64)
    best_dist: list[dict[int, float]] = [dict() for _ in range(n_regions)]

    heap: list[tuple[float, int, int, float]] = []  # (key, face_id, region_id, dist)

    def push(region_id: int, face_id: int, dist: float) -> None:
        prev = best_dist[region_id].get(face_id)
        if prev is not None and dist >= prev:
            return
        best_dist[region_id][face_id] = dist
        key = dist * _capacity_factor(region_area[region_id], abar, zeta, alpha_per_region[region_id])
        heapq.heappush(heap, (key, face_id, region_id, dist))

    for region_id, seed_face in enumerate(region_seed):
        label[seed_face] = region_id
        region_area[region_id] = float(working.face_areas[seed_face])
        best_dist[region_id][seed_face] = 0.0

    for region_id, seed_face in enumerate(region_seed):
        for neighbor, cost, _is_bridge in dual_graph.adjacency[seed_face]:
            if label[neighbor] == -1:
                push(region_id, neighbor, cost)

    claims: list[tuple[int, int, float]] = []
    stale_requeues = 0

    while heap:
        key, face_id, region_id, dist = heapq.heappop(heap)
        if label[face_id] != -1:
            continue
        if best_dist[region_id].get(face_id) != dist:
            continue  # superseded by a shorter path for this (region, face) pair

        current_key = dist * _capacity_factor(region_area[region_id], abar, zeta, alpha_per_region[region_id])
        if current_key > key:
            stale_requeues += 1
            heapq.heappush(heap, (current_key, face_id, region_id, dist))
            continue

        label[face_id] = region_id
        region_area[region_id] += float(working.face_areas[face_id])
        claims.append((face_id, region_id, key))

        for neighbor, cost, _is_bridge in dual_graph.adjacency[face_id]:
            if label[neighbor] == -1:
                push(region_id, neighbor, dist + cost)

    return GrowthResult(
        label=label,
        region_area=region_area,
        region_seed=region_seed,
        claims=claims,
        stale_requeues=stale_requeues,
    )

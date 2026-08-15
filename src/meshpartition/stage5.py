"""Stage 5 -- Seeding.

Place n_k deterministic seeds per major component using graph-metric
farthest-point sampling (multi-source Dijkstra): the first seed is the
component's lowest-id face (stable-index tie-break, A9), and each subsequent
seed is whichever remaining face is farthest (in dual-graph distance) from
every seed chosen so far, ties again broken by lowest face id.

Piece counts are re-apportioned here rather than reused verbatim from Stage 1.
Stage 1's satellite_host is only a provisional nearest-centroid guess made
before any bridge exists. Stage 2 may resolve a different, real anchor for an
adopted satellite, or promote a satellite with no valid bridge into its own
major component -- either changes how much area is actually "behind" each
major. Re-running apportionment against Stage 2's real majors/anchors (using
Stage 1's own _apportion helper) keeps the total seed count equal to the
originally requested N while giving a promoted component its fair share
instead of zero -- or raises, mirroring Stage 1's own guard, if promotion
left more majors than N to go around (see _reapportion).

Seeds are placed within a major component's own induced subgraph only --
walking through a bridge edge into a satellite (or another major) would place
a seed on geometry that never grows independently of its anchor, per Stage 7.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .stage1 import TriageResult, _apportion
from .stage2 import BridgeResult
from .stage4 import DualGraph


@dataclass
class SeedResult:
    piece_counts: dict[int, int]  # major component id -> seed count, post Stage-2 reconciliation
    seeds: dict[int, list[int]]  # major component id -> ordered list of seed face ids
    seed_owner: dict[int, int]  # face id -> owning major component id, for every seed


def _reapportion(triage: TriageResult, bridges: BridgeResult) -> dict[int, int]:
    adopted_area: dict[int, float] = {c: 0.0 for c in bridges.major_ids}
    for satellite, anchor in bridges.anchors.items():
        adopted_area[anchor.other_component] += triage.component_areas[satellite]

    weights = {c: triage.component_areas[c] + adopted_area[c] for c in bridges.major_ids}
    n_pieces = sum(triage.piece_counts.values())
    if n_pieces < len(bridges.major_ids):
        raise ValueError(
            f"N={n_pieces} is too low: {len(bridges.major_ids)} components require their own "
            f"piece after bridging ({len(bridges.promoted_ids)} satellite(s) found no valid "
            "bridge and were promoted to major), so at least that many pieces are required."
        )
    return _apportion(sorted(bridges.major_ids), weights, n_pieces)


def _induced_adjacency(
    dual_graph: DualGraph, face_component_id: np.ndarray, major: int
) -> dict[int, list[tuple[int, float]]]:
    face_set = {int(f) for f in np.nonzero(face_component_id == major)[0]}
    adjacency: dict[int, list[tuple[int, float]]] = {f: [] for f in face_set}
    for f in face_set:
        for neighbor, cost, _is_bridge in dual_graph.adjacency[f]:
            if neighbor in face_set:
                adjacency[f].append((neighbor, cost))
    return adjacency


def _multi_source_distances(
    adjacency: dict[int, list[tuple[int, float]]], sources: list[int]
) -> dict[int, float]:
    dist: dict[int, float] = {f: math.inf for f in adjacency}
    heap: list[tuple[float, int]] = []
    for s in sources:
        dist[s] = 0.0
        heapq.heappush(heap, (0.0, s))

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, cost in adjacency[u]:
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _farthest_point_seed(adjacency: dict[int, list[tuple[int, float]]], n_seeds: int) -> list[int]:
    faces = sorted(adjacency.keys())
    if n_seeds <= 0 or not faces:
        return []

    seeds = [faces[0]]
    chosen = {faces[0]}
    while len(seeds) < n_seeds:
        remaining = [f for f in faces if f not in chosen]
        if not remaining:
            break
        dist = _multi_source_distances(adjacency, seeds)
        next_seed = max(remaining, key=lambda f: (dist[f], -f))
        seeds.append(next_seed)
        chosen.add(next_seed)
    return seeds


def seed(
    dual_graph: DualGraph,
    triage: TriageResult,
    bridges: BridgeResult,
) -> SeedResult:
    piece_counts = _reapportion(triage, bridges)

    seeds: dict[int, list[int]] = {}
    seed_owner: dict[int, int] = {}
    for major in sorted(bridges.major_ids):
        adjacency = _induced_adjacency(dual_graph, triage.face_component_id, major)
        component_seeds = _farthest_point_seed(adjacency, piece_counts[major])
        seeds[major] = component_seeds
        for face_id in component_seeds:
            seed_owner[face_id] = major

    return SeedResult(piece_counts=piece_counts, seeds=seeds, seed_owner=seed_owner)

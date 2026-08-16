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

The actual queue/relaxation loop is a numba kernel (_grow_kernel below), not
Python's heapq: this function is re-run in full by Stage 8's relax() on
every Lloyd iteration (up to i_max times), and pure-Python heapq + dict
overhead measured ~31s on a 200k-face mesh per pass -- dominating relax's
total time far more than Stage 8's own medoid computation. The kernel's
manual array-backed binary heap (four parallel arrays: key, face_id,
region_id, dist) compares entries in that exact tuple order, replicating
Python's default tuple comparison so the popped-key monotonicity and
tie-break behavior -- and therefore the resulting partition -- are unchanged
from the original heapq version.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from .mesh import WorkingMesh
from .stage4 import DualGraph
from .stage5 import SeedResult
from .util import dual_graph_csr

DEFAULT_ALPHA = 1.5
DEFAULT_ZETA = 0.1


@dataclass
class GrowthResult:
    label: np.ndarray  # (F,) int64, region id per face
    region_area: np.ndarray  # (R,) float64, final claimed area per region
    region_seed: list[int]  # region id -> seed face id
    claims: list[tuple[int, int, float]] = field(default_factory=list)  # (face_id, region_id, key), in claim order
    stale_requeues: int = 0  # number of times a popped entry's key was found stale and re-pushed


@njit(cache=True, inline="always")
def _capacity_factor_nb(area: float, abar: float, zeta: float, alpha: float) -> float:
    base = area if area > zeta * abar else zeta * abar
    return (base / abar) ** alpha


@njit(cache=True, inline="always")
def _heap_less(ak, af, ar, ad, bk, bf, br, bd) -> bool:
    # Tuple-order comparison, matching Python's default comparison of
    # (key, face_id, region_id, dist) tuples in the original heapq version.
    if ak != bk:
        return ak < bk
    if af != bf:
        return af < bf
    if ar != br:
        return ar < br
    return ad < bd


@njit(cache=True)
def _heap_sift_up(hk, hf, hr, hd, i):
    while i > 0:
        parent = (i - 1) // 2
        if not _heap_less(hk[i], hf[i], hr[i], hd[i], hk[parent], hf[parent], hr[parent], hd[parent]):
            break
        hk[i], hk[parent] = hk[parent], hk[i]
        hf[i], hf[parent] = hf[parent], hf[i]
        hr[i], hr[parent] = hr[parent], hr[i]
        hd[i], hd[parent] = hd[parent], hd[i]
        i = parent


@njit(cache=True)
def _heap_sift_down(hk, hf, hr, hd, size, i):
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < size and _heap_less(hk[left], hf[left], hr[left], hd[left], hk[smallest], hf[smallest], hr[smallest], hd[smallest]):
            smallest = left
        if right < size and _heap_less(hk[right], hf[right], hr[right], hd[right], hk[smallest], hf[smallest], hr[smallest], hd[smallest]):
            smallest = right
        if smallest == i:
            break
        hk[i], hk[smallest] = hk[smallest], hk[i]
        hf[i], hf[smallest] = hf[smallest], hf[i]
        hr[i], hr[smallest] = hr[smallest], hr[i]
        hd[i], hd[smallest] = hd[smallest], hd[i]
        i = smallest


@njit(cache=True)
def _grow_heap_arrays(hk, hf, hr, hd, cap):
    new_cap = cap * 2
    new_hk = np.empty(new_cap, dtype=np.float64)
    new_hf = np.empty(new_cap, dtype=np.int64)
    new_hr = np.empty(new_cap, dtype=np.int64)
    new_hd = np.empty(new_cap, dtype=np.float64)
    new_hk[:cap] = hk
    new_hf[:cap] = hf
    new_hr[:cap] = hr
    new_hd[:cap] = hd
    return new_hk, new_hf, new_hr, new_hd, new_cap


@njit(cache=True)
def _heap_push(hk, hf, hr, hd, cap, size, key, face, region, dist):
    # total_directed_edges + n_regions (the caller's starting capacity) is an
    # exact bound on ordinary relaxation pushes, but stale requeues
    # (region_area keeps growing, so a popped entry's key can be recomputed
    # and re-pushed) aren't bounded by that count -- grow on demand rather
    # than trying to size for the theoretical worst case up front.
    if size >= cap:
        hk, hf, hr, hd, cap = _grow_heap_arrays(hk, hf, hr, hd, cap)
    hk[size] = key
    hf[size] = face
    hr[size] = region
    hd[size] = dist
    _heap_sift_up(hk, hf, hr, hd, size)
    return hk, hf, hr, hd, cap, size + 1


@njit(cache=True)
def _heap_pop(hk, hf, hr, hd, size):
    key = hk[0]
    face = hf[0]
    region = hr[0]
    dist = hd[0]
    size -= 1
    hk[0] = hk[size]
    hf[0] = hf[size]
    hr[0] = hr[size]
    hd[0] = hd[size]
    _heap_sift_down(hk, hf, hr, hd, size, 0)
    return key, face, region, dist, size


@njit(cache=True)
def _grow_kernel(indptr, nbr, cost, face_areas, region_seed, alpha_per_region, abar, zeta):
    n_faces = face_areas.shape[0]
    n_regions = region_seed.shape[0]

    label = np.full(n_faces, -1, dtype=np.int64)
    region_area = np.zeros(n_regions, dtype=np.float64)
    best_dist = np.full((n_regions, n_faces), np.inf, dtype=np.float64)

    heap_cap = indptr[n_faces] + n_regions + 1
    hk = np.empty(heap_cap, dtype=np.float64)
    hf = np.empty(heap_cap, dtype=np.int64)
    hr = np.empty(heap_cap, dtype=np.int64)
    hd = np.empty(heap_cap, dtype=np.float64)
    heap_size = 0

    claim_face = np.empty(n_faces, dtype=np.int64)
    claim_region = np.empty(n_faces, dtype=np.int64)
    claim_key = np.empty(n_faces, dtype=np.float64)
    n_claims = 0
    stale_requeues = 0

    # Seed labels are set for every region before any neighbor is pushed, so
    # two adjacent seeds never push into each other.
    for region_id in range(n_regions):
        seed_face = region_seed[region_id]
        label[seed_face] = region_id
        region_area[region_id] = face_areas[seed_face]
        best_dist[region_id, seed_face] = 0.0

    for region_id in range(n_regions):
        seed_face = region_seed[region_id]
        for e in range(indptr[seed_face], indptr[seed_face + 1]):
            neighbor = nbr[e]
            if label[neighbor] == -1:
                dist = cost[e]
                if dist < best_dist[region_id, neighbor]:
                    best_dist[region_id, neighbor] = dist
                    key = dist * _capacity_factor_nb(region_area[region_id], abar, zeta, alpha_per_region[region_id])
                    hk, hf, hr, hd, heap_cap, heap_size = _heap_push(
                        hk, hf, hr, hd, heap_cap, heap_size, key, neighbor, region_id, dist
                    )

    while heap_size > 0:
        key, face_id, region_id, dist, heap_size = _heap_pop(hk, hf, hr, hd, heap_size)

        if label[face_id] != -1:
            continue
        if best_dist[region_id, face_id] != dist:
            continue  # superseded by a shorter path for this (region, face) pair

        current_key = dist * _capacity_factor_nb(region_area[region_id], abar, zeta, alpha_per_region[region_id])
        if current_key > key:
            stale_requeues += 1
            hk, hf, hr, hd, heap_cap, heap_size = _heap_push(
                hk, hf, hr, hd, heap_cap, heap_size, current_key, face_id, region_id, dist
            )
            continue

        label[face_id] = region_id
        region_area[region_id] += face_areas[face_id]
        claim_face[n_claims] = face_id
        claim_region[n_claims] = region_id
        claim_key[n_claims] = key
        n_claims += 1

        for e in range(indptr[face_id], indptr[face_id + 1]):
            neighbor = nbr[e]
            if label[neighbor] == -1:
                ndist = dist + cost[e]
                if ndist < best_dist[region_id, neighbor]:
                    best_dist[region_id, neighbor] = ndist
                    nkey = ndist * _capacity_factor_nb(region_area[region_id], abar, zeta, alpha_per_region[region_id])
                    hk, hf, hr, hd, heap_cap, heap_size = _heap_push(
                        hk, hf, hr, hd, heap_cap, heap_size, nkey, neighbor, region_id, ndist
                    )

    return label, region_area, claim_face[:n_claims], claim_region[:n_claims], claim_key[:n_claims], stale_requeues


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
    alpha_per_region = np.ascontiguousarray(
        np.broadcast_to(np.asarray(alpha, dtype=np.float64), (n_regions,))
    )
    region_seed_arr = np.asarray(region_seed, dtype=np.int64)

    indptr, nbr, cost = dual_graph_csr(dual_graph)

    face_areas = np.ascontiguousarray(working.face_areas, dtype=np.float64)
    label, region_area, claim_face, claim_region, claim_key, stale_requeues = _grow_kernel(
        indptr, nbr, cost, face_areas, region_seed_arr, alpha_per_region, float(abar), float(zeta)
    )

    claims = list(zip(claim_face.tolist(), claim_region.tolist(), claim_key.tolist()))

    return GrowthResult(
        label=label,
        region_area=region_area,
        region_seed=region_seed,
        claims=claims,
        stale_requeues=int(stale_requeues),
    )

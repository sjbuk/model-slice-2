"""Stage 4 -- Dual graph construction.

    1. Node weight = face area.
    2. Surface edge cost = ||ci - cj|| x (1 + mu * concavity(e)).
    3. Insert bridge edges from Stage 2.

Concavity sign convention: for a solid, every other face's geometry lies on
the inward (negative-normal) side of any given face's own supporting plane --
that is the defining property of convexity. So for edge e shared by faces a
and b, take a point p on e and the opposite vertex r_b of b (the corner of b
not on e): if r_b lies on the OUTWARD side of a's plane (dot(n_a, r_b - p) > 0),
the fold pokes outward past a's boundary -- a valley/concave crease relative
to the solid. If it lies on the inward side (<= 0), the fold is convex (or
flat, at exactly 0). This was calibrated against Fixture F's real cube corner
(known convex by construction) before trusting the sign both ways.

Concavity is computed once per undirected edge, from a fixed reference face
(the lower face id), and that single value is reused for both directions of
the dual graph -- so cost symmetry (GI-4) holds by construction, not by
relying on the concavity formula itself being perspective-symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .mesh import WorkingMesh
from .stage2 import BridgeResult
from .util import face_normals

DEFAULT_MU = 2.0
DEFAULT_LAMBDA = 5.0
MIN_EDGE_COST = 1e-12


@dataclass
class DualEdge:
    face_a: int
    face_b: int
    cost: float
    is_bridge: bool
    concavity: float = 0.0


@dataclass
class DualGraph:
    node_weights: np.ndarray  # (F,) face areas
    adjacency: dict[int, list[tuple[int, float, bool]]] = field(default_factory=dict)
    edges: list[DualEdge] = field(default_factory=list)


def _opposite_vertex_position(working: WorkingMesh, face_id: int, edge_key: tuple[int, int]):
    pos = working.faces_pos[face_id]
    edge_set = set(edge_key)
    for idx in range(3):
        if int(pos[idx]) not in edge_set:
            return working.positions[pos[idx]]
    raise ValueError(f"face {face_id} does not contain edge {edge_key}")


def concavity(
    working: WorkingMesh,
    normals: np.ndarray,
    edge_key: tuple[int, int],
    face_a: int,
    face_b: int,
) -> float:
    p = working.positions[edge_key[0]]
    n_a = normals[face_a]
    r_b = _opposite_vertex_position(working, face_b, edge_key)
    signed = float(np.dot(n_a, r_b - p))
    if signed <= 0.0:
        return 0.0
    bend = 1.0 - float(np.dot(n_a, normals[face_b]))
    return max(bend, 0.0)


def _surface_edge_cost(
    working: WorkingMesh,
    normals: np.ndarray,
    edge_key: tuple[int, int],
    face_a: int,
    face_b: int,
    mu: float,
) -> float:
    dist = float(np.linalg.norm(working.face_centroids[face_a] - working.face_centroids[face_b]))
    conc = concavity(working, normals, edge_key, face_a, face_b)
    return max(dist * (1.0 + mu * conc), MIN_EDGE_COST), conc


def build_dual_graph(
    working: WorkingMesh,
    bridges: BridgeResult | None = None,
    mu: float = DEFAULT_MU,
    lam: float = DEFAULT_LAMBDA,
) -> DualGraph:
    normals = face_normals(working.face_corner_positions())
    adjacency: dict[int, list[tuple[int, float, bool]]] = {
        i: [] for i in range(working.face_count)
    }
    edges: list[DualEdge] = []

    def add_edge(a: int, b: int, cost: float, is_bridge: bool, conc: float = 0.0) -> None:
        adjacency[a].append((b, cost, is_bridge))
        adjacency[b].append((a, cost, is_bridge))
        edges.append(DualEdge(a, b, cost, is_bridge, conc))

    for edge_key, faces in working.edge_faces.items():
        if len(faces) < 2:
            continue
        for face_a, face_b in combinations(sorted(faces), 2):
            cost, conc = _surface_edge_cost(working, normals, edge_key, face_a, face_b, mu)
            add_edge(face_a, face_b, cost, is_bridge=False, conc=conc)

    if bridges is not None:
        for edge_list in bridges.bridges.values():
            for bridge in edge_list:
                cost = max(lam * bridge.distance, MIN_EDGE_COST)
                add_edge(bridge.satellite_face, bridge.other_face, cost, is_bridge=True)

    return DualGraph(node_weights=working.face_areas.copy(), adjacency=adjacency, edges=edges)

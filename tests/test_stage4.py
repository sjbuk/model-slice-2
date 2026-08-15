"""Stage 4 validation, per docs/mesh_partitioning_test_spec_v3.md."""

from __future__ import annotations

import numpy as np
import pytest

from meshpartition.mesh import RawMesh
from meshpartition.stage0 import ingest
from meshpartition.stage1 import triage
from meshpartition.stage2 import build_bridges
from meshpartition.stage4 import build_dual_graph
from meshpartition.util import UnionFind

from fixtures import ALL_FIXTURES, fixture_c_shirt


def _build_graph(fixture_fn, n_pieces=5):
    working, _ = ingest(fixture_fn())
    result = triage(working, n_pieces=n_pieces)
    bridges = build_bridges(working, result)
    graph = build_dual_graph(working, bridges)
    return working, graph


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_4_1_symmetry(name):
    """4.1 -- every edge U->V has a matching V->U entry with exactly equal cost."""
    _, graph = _build_graph(ALL_FIXTURES[name])

    for edge in graph.edges:
        forward = [c for (n, c, _) in graph.adjacency[edge.face_a] if n == edge.face_b]
        backward = [c for (n, c, _) in graph.adjacency[edge.face_b] if n == edge.face_a]
        assert forward and backward
        assert forward[0] == backward[0] == edge.cost


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_4_2_positive_costs(name):
    """4.2 -- no zero, negative, NaN, or infinite edge cost."""
    _, graph = _build_graph(ALL_FIXTURES[name])

    assert len(graph.edges) > 0
    costs = np.array([e.cost for e in graph.edges])
    assert np.all(np.isfinite(costs))
    assert np.all(costs > 0)


def _two_triangle_mesh(opposite_b: list[float]) -> RawMesh:
    """Two triangles sharing edge (p1,p2). face_a = [p0,p1,p2] is fixed, with
    normal +x -- this is exactly Fixture F's own +X cube-face triangle
    (hand-verified convex against its +Z neighbor). face_b shares the same
    edge but its opposite vertex is `opposite_b`, wound so its own normal is
    a genuine outward normal for that triangle.

    Winding face_b's *existing* vertices the other way around would NOT
    produce a concave fold here -- that only flips which way its declared
    normal points while every vertex stays exactly where it was, so the sign
    test below (which depends on where the opposite vertex actually sits
    relative to face_a's plane, not on face_b's own winding) sees the same
    geometry either way. A genuinely concave fold requires moving the
    opposite vertex to the other side of face_a's plane -- that's what
    `opposite_b` does: the fixture's own point (-h,-h,h) sits on the inward
    (convex) side of face_a's x=h plane; a point with x > h sits on the
    outward (concave) side of that same plane.
    """
    h = 0.5
    positions = np.array(
        [
            [h, -h, -h],  # p0: face_a's opposite vertex
            [h, h, h],  # p1: shared edge endpoint
            [h, -h, h],  # p2: shared edge endpoint
            opposite_b,  # p3: face_b's opposite vertex
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.int64)
    return RawMesh(positions=positions, faces_pos=faces)


def test_4_3_concavity_weighting():
    """4.3 -- a concave crease costs strictly more to cross than a flat/convex one.

    Fixture F's cube corners are, by construction, convex (a real solid box
    viewed from outside) -- there is no concave edge in it to test against.
    This starts from that exact geometry (face_a is Fixture F's own +X
    triangle) and repositions only face_b's opposite vertex to the other
    side of face_a's plane, producing the genuinely concave version of a
    90-degree fold to compare against.
    """
    h = 0.5
    convex_working, _ = ingest(_two_triangle_mesh([-h, -h, h]))  # Fixture F's real +Z corner
    concave_working, _ = ingest(_two_triangle_mesh([2 * h, -h, h]))  # mirrored to the outward side

    convex_graph = build_dual_graph(convex_working)
    concave_graph = build_dual_graph(concave_working)

    assert len(convex_graph.edges) == 1
    assert len(concave_graph.edges) == 1

    convex_edge = convex_graph.edges[0]
    concave_edge = concave_graph.edges[0]

    assert convex_edge.concavity == 0.0
    assert concave_edge.concavity > 0.0
    assert concave_edge.cost > convex_edge.cost


def test_4_4_augmented_connectivity():
    """4.4 -- the dual graph, with bridges, is a single connected component."""
    working, _ = ingest(fixture_c_shirt())
    result = triage(working, n_pieces=5)
    bridges = build_bridges(working, result)
    graph = build_dual_graph(working, bridges)

    uf = UnionFind(working.face_count)
    for edge in graph.edges:
        uf.union(edge.face_a, edge.face_b)

    roots = {uf.find(i) for i in range(working.face_count)}
    assert len(roots) == 1

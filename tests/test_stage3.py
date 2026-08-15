"""Stage 3 validation, per docs/mesh_partitioning_test_spec_v3.md."""

from __future__ import annotations

import numpy as np

from meshpartition.stage0 import ingest
from meshpartition.stage3 import refine

from fixtures import fixture_a_sphere, fixture_f_uvbox


def test_3_1_max_area_cap():
    """3.1 -- with an artificially small max area, no face in the result exceeds it."""
    working, _ = ingest(fixture_a_sphere())
    mean_area = working.total_area() / working.face_count
    small_a_max = mean_area * 0.4

    refined, stats = refine(working, abar=small_a_max, a_max_fraction=1.0)

    assert stats.faces_split > 0
    assert np.all(refined.face_areas <= small_a_max + 1e-12)


def test_3_2_t_junction_prevention():
    """3.2 -- no newly created edge has more than 2 incident faces."""
    working, _ = ingest(fixture_a_sphere())
    mean_area = working.total_area() / working.face_count

    refined, _ = refine(working, abar=mean_area * 0.4, a_max_fraction=1.0)

    incident_counts = [len(faces) for faces in refined.edge_faces.values()]
    assert max(incident_counts) <= 2
    # The sphere started fully watertight/manifold; bisection must preserve that.
    assert len(refined.boundary_edges) == 0
    assert all(count == 2 for count in incident_counts)


def test_3_3_strict_geometry():
    """3.3 -- total area is exactly conserved; no Laplacian smoothing."""
    working, _ = ingest(fixture_a_sphere())
    mean_area = working.total_area() / working.face_count

    refined, stats = refine(working, abar=mean_area * 0.4, a_max_fraction=1.0)

    assert stats.faces_split > 0
    relative_delta = abs(refined.total_area() - working.total_area()) / working.total_area()
    assert relative_delta < 1e-9


def test_3_4_attribute_interpolation():
    """3.4 -- new bisection-vertex UVs are the exact midpoint of the parent face's own UVs."""
    working, _ = ingest(fixture_f_uvbox())

    # Face 0 (first triangle of the +X face) has corner UVs (0,0),(1,0),(1,1).
    # Its longest edge is the diagonal between corners 0 and 2, so bisecting it
    # must produce a UV entry at exactly the midpoint (0.5, 0.5).
    refined, stats = refine(working, abar=0.1, a_max_fraction=1.0)

    assert stats.faces_split > 0
    assert np.any(np.all(np.isclose(refined.uvs, [0.5, 0.5]), axis=1))

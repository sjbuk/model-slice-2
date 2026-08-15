"""Stage 0 validation, per docs/mesh_partitioning_test_spec_v3.md."""

from __future__ import annotations

import numpy as np
import pytest

from meshpartition.stage0 import default_tolerances, ingest
from meshpartition.util import triangle_areas

from fixtures import ALL_FIXTURES, fixture_b_soup, fixture_f_uvbox


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_0_1_area_conservation(name):
    """0.1 -- sum of face areas before/after conditioning must match within 1e-6 relative."""
    raw = ALL_FIXTURES[name]()
    working, stats = ingest(raw)

    denom = max(abs(stats.area_before), 1e-12)
    relative_delta = abs(stats.area_after - stats.area_before) / denom
    assert relative_delta < 1e-6

    # Cross-check against an independent area computation on the working mesh.
    recomputed = float(np.sum(triangle_areas(working.face_corner_positions())))
    assert abs(recomputed - working.total_area()) < 1e-9


def test_0_2_degenerate_purge():
    """0.2 -- degenerate faces present pre-ingest, exactly zero post-ingest."""
    raw = fixture_b_soup()
    _, eps = default_tolerances(raw.positions)
    areas_before = triangle_areas(raw.face_corner_positions())
    degenerate_before = int(np.sum(areas_before < eps))
    assert degenerate_before > 0, "fixture B must contain at least one degenerate face"

    working, stats = ingest(raw)
    degenerate_after = int(np.sum(working.face_areas < stats.epsilon_area))
    assert degenerate_after == 0
    assert stats.degenerate_face_count == degenerate_before


def _uv_island_count(faces_uv: np.ndarray) -> int:
    """Union-find over faces that share a UV corner index."""
    parent = list(range(len(faces_uv)))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    uv_to_face: dict[int, int] = {}
    for face_id, corners in enumerate(faces_uv):
        for uv_idx in corners:
            uv_idx = int(uv_idx)
            if uv_idx in uv_to_face:
                union(face_id, uv_to_face[uv_idx])
            else:
                uv_to_face[uv_idx] = face_id

    return len({find(i) for i in range(len(faces_uv))})


def _hard_edge_count(working) -> int:
    face_normal_id = working.faces_normal[:, 0]
    count = 0
    for key, incident in working.edge_faces.items():
        if len(incident) == 2:
            f0, f1 = incident
            if face_normal_id[f0] != face_normal_id[f1]:
                count += 1
    return count


def test_0_3_attribute_survival():
    """0.3 -- UV islands and hard edges must be identical before/after position welding."""
    raw = fixture_f_uvbox()

    islands_before = _uv_island_count(raw.faces_uv)
    assert islands_before == 6  # one per cube face, by construction

    working, _ = ingest(raw)

    assert working.vertex_count == 8, "welding should collapse 24 authored corners to 8 cube corners"
    islands_after = _uv_island_count(working.faces_uv)
    assert islands_after == islands_before

    assert _hard_edge_count(working) == 12  # the 12 true cube edges


def test_0_4_non_manifold_survival():
    """0.4 -- half-edge build must not crash; T-junction recorded as a 3-face edge."""
    raw = fixture_b_soup()

    working, _ = ingest(raw)  # must not raise

    assert len(working.nonmanifold_edges) > 0
    tjunction_edge = next(iter(working.nonmanifold_edges))
    incident_faces = working.edge_faces[tjunction_edge]
    assert len(incident_faces) == 3
    assert len(set(incident_faces)) == 3  # distinct face ids, not duplicates


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_0_5_determinism(name):
    """0.5 -- loading/ingesting the same fixture twice must yield byte-identical output."""
    generator = ALL_FIXTURES[name]

    working_a, _ = ingest(generator())
    working_b, _ = ingest(generator())

    assert np.array_equal(working_a.positions, working_b.positions)
    assert np.array_equal(working_a.faces_pos, working_b.faces_pos)
    assert np.array_equal(working_a.face_areas, working_b.face_areas)
    assert np.array_equal(working_a.face_centroids, working_b.face_centroids)

"""Stage 1 validation, per docs/mesh_partitioning_test_spec_v3.md."""

from __future__ import annotations

import numpy as np
import pytest

from meshpartition.mesh import WorkingMesh
from meshpartition.stage0 import ingest
from meshpartition.stage1 import _component_centroid, triage

from fixtures import fixture_a_sphere, fixture_c_shirt, fixture_c_shirt_two_torsos


def test_1_1_classification():
    """1.1 -- the torso is Major, buttons are Satellite."""
    working, _ = ingest(fixture_c_shirt())
    result = triage(working, n_pieces=5)

    assert len(result.major_ids) == 1  # the torso
    assert len(result.satellite_ids) == 3  # three buttons

    torso = result.major_ids[0]
    for satellite in result.satellite_ids:
        assert result.component_areas[satellite] < result.component_areas[torso]
        assert result.satellite_host[satellite] == torso


def test_1_2_allocation_math():
    """1.2 -- a single major component with N=5 receives exactly 5 pieces."""
    working, _ = ingest(fixture_a_sphere())
    result = triage(working, n_pieces=5)

    assert len(result.major_ids) == 1
    only_major = result.major_ids[0]
    assert result.piece_counts[only_major] == 5
    assert sum(result.piece_counts.values()) == 5


def test_1_3_infeasibility_catch():
    """1.3 -- requesting fewer pieces than major components raises an error."""
    working, _ = ingest(fixture_c_shirt_two_torsos())

    with pytest.raises(ValueError, match="too low"):
        triage(working, n_pieces=1)


def test_1_4_degenerate_component_centroid_no_crash():
    """A component whose faces are all zero-area (e.g. slivers that weld
    collapses to nothing on a dense, large mesh) must still yield a centroid
    instead of numpy's 'Weights sum to zero, can't be normalized' ZeroDivisionError.
    """
    faces_pos = np.array([[0, 0, 1]], dtype=np.int64)  # degenerate: repeated vertex
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    working = WorkingMesh(
        positions=positions,
        faces_pos=faces_pos,
        uvs=None,
        faces_uv=None,
        normals=None,
        faces_normal=None,
        material_ids=np.array([0]),
        face_areas=np.array([0.0]),
        face_centroids=np.array([[0.5, 0.0, 0.0]]),
        edge_faces={},
        boundary_edges=frozenset(),
        nonmanifold_edges=frozenset(),
        vertex_source_groups=[[0], [1]],
        source_face_indices=np.array([0]),
    )
    centroid = _component_centroid(working, np.array([0]), 0)
    np.testing.assert_allclose(centroid, [0.5, 0.0, 0.0])

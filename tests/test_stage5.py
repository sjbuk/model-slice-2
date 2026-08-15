"""Stage 5 validation, per docs/mesh_partitioning_test_spec_v3.md."""

from __future__ import annotations

import numpy as np
import pytest

from meshpartition.mesh import RawMesh
from meshpartition.stage0 import ingest
from meshpartition.stage1 import triage
from meshpartition.stage2 import build_bridges
from meshpartition.stage4 import build_dual_graph
from meshpartition.stage5 import seed

from fixtures import ALL_FIXTURES, fixture_a_sphere, fixture_c_shirt

# Kept small enough that every fixture's smallest major component (Fixture
# D's two-triangle decoy floor) still has enough distinct faces to seat its
# apportioned share.
DEFAULT_N_PIECES = 3


def _seed_fixture(fixture_fn, n_pieces=DEFAULT_N_PIECES):
    working, _ = ingest(fixture_fn())
    result = triage(working, n_pieces=n_pieces)
    bridges = build_bridges(working, result)
    graph = build_dual_graph(working, bridges)
    return seed(graph, result, bridges), n_pieces


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_5_1_count(name):
    """5.1 -- total seed count across all major components matches N exactly."""
    seed_result, n_pieces = _seed_fixture(ALL_FIXTURES[name])

    total_seeds = sum(len(faces) for faces in seed_result.seeds.values())
    assert total_seeds == n_pieces
    assert sum(seed_result.piece_counts.values()) == n_pieces


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_5_2_distinctness(name):
    """5.2 -- no seed cell is reused, within or across major components."""
    seed_result, _ = _seed_fixture(ALL_FIXTURES[name])

    all_seed_faces = [f for faces in seed_result.seeds.values() for f in faces]
    assert len(all_seed_faces) == len(set(all_seed_faces))


def test_5_3_determinism():
    """5.3 -- ten runs with identical parameters yield identical seed indices."""
    runs = [_seed_fixture(fixture_a_sphere)[0] for _ in range(10)]

    first = runs[0].seeds
    for other in runs[1:]:
        assert other.seeds == first


def test_5_4_infeasibility_after_promotion():
    """5.4 -- when Stage 2 promotes enough satellites that majors outnumber N,
    seeding raises rather than silently returning more pieces than requested
    (Stage 1's own too-few-N guard only sees pre-promotion majors, so this
    has to be re-checked here once promotion is known).
    """
    working, _ = ingest(fixture_c_shirt())
    result = triage(working, n_pieces=2)
    assert len(result.major_ids) == 1  # only the torso -- fits under N=2 pre-promotion

    # Move every button far enough away (in distinct directions, so none can
    # bridge to another moved button's new neighborhood either) that none
    # finds a valid bridge back to the torso.
    far_positions = working.positions.copy()
    for offset, satellite in enumerate(result.satellite_ids, start=1):
        sat_verts = np.unique(working.faces_pos[result.face_component_id == satellite])
        far_positions[sat_verts] += np.array([10.0 * offset, 0.0, 0.0])

    moved_working, _ = ingest(RawMesh(positions=far_positions, faces_pos=working.faces_pos))
    moved_result = triage(moved_working, n_pieces=2)
    bridges = build_bridges(moved_working, moved_result)
    assert len(bridges.promoted_ids) == 3
    assert len(bridges.major_ids) == 4  # torso + 3 promoted buttons, > N=2

    graph = build_dual_graph(moved_working, bridges)
    with pytest.raises(ValueError, match="too low"):
        seed(graph, moved_result, bridges)

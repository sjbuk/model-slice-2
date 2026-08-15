"""Stage 5 validation, per docs/mesh_partitioning_test_spec_v3.md."""

from __future__ import annotations

import pytest

from meshpartition.stage0 import ingest
from meshpartition.stage1 import triage
from meshpartition.stage2 import build_bridges
from meshpartition.stage4 import build_dual_graph
from meshpartition.stage5 import seed

from fixtures import ALL_FIXTURES, fixture_a_sphere

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

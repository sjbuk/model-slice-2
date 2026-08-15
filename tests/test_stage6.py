"""Stage 6 validation, per docs/mesh_partitioning_test_spec_v3.md."""

from __future__ import annotations

import numpy as np
import pytest

from meshpartition.stage0 import ingest
from meshpartition.stage1 import triage
from meshpartition.stage2 import build_bridges
from meshpartition.stage4 import build_dual_graph
from meshpartition.stage5 import seed
from meshpartition.stage6 import grow

from fixtures import ALL_FIXTURES, fixture_a_sphere, fixture_e_ribbon

# Matches Stage 5's tests: small enough that every fixture's smallest major
# component can still seat its apportioned seed count.
DEFAULT_N_PIECES = 3


def _run_growth(fixture_fn, n_pieces=DEFAULT_N_PIECES):
    working, _ = ingest(fixture_fn())
    result = triage(working, n_pieces=n_pieces)
    bridges = build_bridges(working, result)
    graph = build_dual_graph(working, bridges)
    seed_result = seed(graph, result, bridges)
    growth = grow(working, graph, seed_result, abar=result.abar)
    return working, growth


def test_6_1_lazy_evaluation_queue():
    """6.1 -- no face is claimed on a stale key: the popped claim-key sequence
    is non-decreasing, exactly Dijkstra's finalized-distance monotonicity
    carried over to the capacity-scaled key. If a stale (too-small) key had
    ever been allowed to win instead of being recomputed and re-pushed, this
    would be violated.
    """
    _, growth = _run_growth(fixture_a_sphere, n_pieces=5)

    assert growth.stale_requeues > 0  # the mechanism actually triggers on this fixture
    keys = [key for (_face, _region, key) in growth.claims]
    assert all(keys[i] <= keys[i + 1] + 1e-12 for i in range(len(keys) - 1))


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_6_2_total_coverage(name):
    """6.2 -- every face is labelled; zero unclaimed cells at exhaustion."""
    working, growth = _run_growth(ALL_FIXTURES[name])

    assert np.count_nonzero(growth.label == -1) == 0
    assert working.face_count == len(growth.label)


def test_6_3_soft_cap_integrity():
    """6.3 -- N=2 on the Ribbon forces one region to over-consume; the
    pipeline still completes cleanly with no orphaned cells.
    """
    working, growth = _run_growth(fixture_e_ribbon, n_pieces=2)

    assert np.count_nonzero(growth.label == -1) == 0
    assert len(growth.region_seed) == 2


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_6_4_area_accounting(name):
    """6.4 -- summed tracked region areas match total mesh area (relative delta < 1e-6)."""
    working, growth = _run_growth(ALL_FIXTURES[name])

    total = working.total_area()
    relative_delta = abs(float(growth.region_area.sum()) - total) / total
    assert relative_delta < 1e-6

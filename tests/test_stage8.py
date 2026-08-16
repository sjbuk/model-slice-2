"""Stage 8 validation, per docs/mesh_partitioning_test_spec_v3.md."""

from __future__ import annotations

import numpy as np

from meshpartition.stage0 import ingest
from meshpartition.stage1 import triage
from meshpartition.stage2 import build_bridges
from meshpartition.stage4 import build_dual_graph
from meshpartition.stage5 import seed
from meshpartition.stage6 import grow
from meshpartition.stage7 import repair
from meshpartition.progress import ProgressReporter
from meshpartition.stage8 import DEFAULT_EPSILON_BAL, _EXACT_MEDOID_MAX_FACES, _medoid, relax

from fixtures import fixture_a_sphere, fixture_e_ribbon


def _run_to_repair(fixture_fn, n_pieces):
    working, _ = ingest(fixture_fn())
    result = triage(working, n_pieces=n_pieces)
    bridges = build_bridges(working, result)
    graph = build_dual_graph(working, bridges)
    seed_result = seed(graph, result, bridges)
    growth = grow(working, graph, seed_result, abar=result.abar)
    repaired = repair(working, graph, result, bridges, growth)
    return working, graph, result, bridges, seed_result, growth, repaired


def test_8_1_convergence():
    """8.1 -- the loop drives max area deviation under the 25% target."""
    working, graph, result, bridges, seed_result, growth, repaired = _run_to_repair(
        fixture_a_sphere, n_pieces=5
    )

    relaxed = relax(working, graph, result, bridges, seed_result, growth, repaired)

    assert relaxed.max_area_deviation <= DEFAULT_EPSILON_BAL
    assert relaxed.converged


def test_8_2_best_retained():
    """8.2 -- the returned state is whichever iteration scored best, even if
    the final iteration scored worse.

    N=18 on the Sphere, capped at i_max=3, is a case where the score dips on
    iteration 2 and then rises again on iteration 3 (the loop stops there
    because i_max was hit, not because it reconverged to something better) --
    so the last iteration is demonstrably not the best one, and the test can
    check the retained result is iteration 2, not iteration 3.
    """
    working, graph, result, bridges, seed_result, growth, repaired = _run_to_repair(
        fixture_a_sphere, n_pieces=18
    )

    relaxed = relax(working, graph, result, bridges, seed_result, growth, repaired, i_max=3)

    assert len(relaxed.scores) == relaxed.iterations_run == 3
    assert relaxed.score == min(relaxed.scores)
    assert relaxed.score != relaxed.scores[-1]  # final iteration was worse than the best


def test_relax_substep_progress_climbs_monotonically_across_iterations():
    """The medoid substep counter reported to `progress` must climb across
    the whole relax call, not reset to 1 at the start of every Lloyd
    iteration -- a per-iteration-only counter looks identical whether it's
    iteration 1 or iteration 14 of i_max, giving no sense of overall
    progress through the phase.
    """
    working, graph, result, bridges, seed_result, growth, repaired = _run_to_repair(
        fixture_a_sphere, n_pieces=18
    )

    snapshots = []
    reporter = ProgressReporter(on_update=snapshots.append, min_interval=0.0)
    relaxed = relax(
        working, graph, result, bridges, seed_result, growth, repaired, i_max=3, progress=reporter
    )
    assert relaxed.iterations_run == 3  # confirms >1 Lloyd pass actually ran

    substep_values = [
        s["substep_current"] for s in snapshots if s.get("substep_total") is not None
    ]
    assert len(substep_values) > 1
    # Non-decreasing across the whole phase (never drops back to a small
    # value partway through, the way a per-iteration-only counter would at
    # the start of iteration 2).
    assert substep_values == sorted(substep_values)
    # Actually increases somewhere (not just flat) -- two Lloyd passes over
    # 18 regions each should produce real forward movement.
    assert substep_values[-1] > substep_values[0]


def test_8_4_large_region_medoid_stays_bounded():
    """Regression test for the web UI near-OOM on CarConcept.glb: exact
    all-pairs Dijkstra materializes a dense n x n distance matrix, which is
    fine at fixture scale but gigabytes for a real hard-surface asset's
    large regions (a force_exact_count run left a 20k+-face region, and the
    dense matrix alone approached the container's memory limit).

    A region past _EXACT_MEDOID_MAX_FACES faces must fall back to the
    sampled-candidate approximation instead of ever building that matrix,
    and still return a face that is actually part of the region. Modeled as
    a plain path graph (node i <-> i+1, cost 1) so the true medoid is known
    (the middle node minimizes total distance along a chain) and the
    approximation's result can be checked for being in the right
    neighborhood, not just "didn't crash".
    """
    n = _EXACT_MEDOID_MAX_FACES + 500
    adjacency = {i: [] for i in range(n)}
    for i in range(n - 1):
        adjacency[i].append((i + 1, 1.0))
        adjacency[i + 1].append((i, 1.0))

    medoid = _medoid(adjacency)

    assert medoid in adjacency
    true_middle = n // 2
    assert abs(medoid - true_middle) < n // 10  # approximate, but in the right neighborhood


def test_8_3_termination_limits():
    """8.3 -- an unbalanceable shape (too many pieces for the Ribbon's face
    count -- +-1 face of quantization is already a huge relative deviation)
    halts strictly at i_max instead of looping forever, and never falsely
    reports convergence.
    """
    working, graph, result, bridges, seed_result, growth, repaired = _run_to_repair(
        fixture_e_ribbon, n_pieces=20
    )

    relaxed = relax(working, graph, result, bridges, seed_result, growth, repaired, i_max=4)

    assert relaxed.iterations_run == 4
    assert not relaxed.converged
    assert relaxed.max_area_deviation > DEFAULT_EPSILON_BAL

"""Stage 7 validation, per docs/mesh_partitioning_test_spec_v3.md."""

from __future__ import annotations

import numpy as np

from meshpartition.stage0 import ingest
from meshpartition.stage1 import triage
from meshpartition.stage2 import build_bridges
from meshpartition.stage4 import build_dual_graph
from meshpartition.stage5 import seed
from meshpartition.stage6 import DEFAULT_ALPHA, grow
from meshpartition.stage7 import DEFAULT_GAMMA, apply_dump, label_components, repair

from fixtures import fixture_a_sphere, fixture_c_shirt, fixture_e_ribbon

DEFAULT_N_PIECES = 3


def _run_pipeline(fixture_fn, n_pieces=DEFAULT_N_PIECES):
    working, _ = ingest(fixture_fn())
    result = triage(working, n_pieces=n_pieces)
    bridges = build_bridges(working, result)
    graph = build_dual_graph(working, bridges)
    seed_result = seed(graph, result, bridges)
    growth = grow(working, graph, seed_result, abar=result.abar)
    repaired = repair(working, graph, result, bridges, growth)
    return working, graph, result, bridges, repaired


def test_7_1_satellite_coherence():
    """7.1 -- a button's label matches its anchor cell's label on the shirt."""
    _, _, _, bridges, repaired = _run_pipeline(fixture_c_shirt)

    assert bridges.anchors  # the shirt fixture must actually produce anchored satellites
    for satellite, anchor in bridges.anchors.items():
        button_face = anchor.satellite_face
        assert repaired.label[button_face] == repaired.label[anchor.other_face]


def test_7_2_connectivity():
    """7.2 -- after repair, every label is exactly one connected component."""
    working, graph, _, _, repaired = _run_pipeline(fixture_a_sphere)

    groups = label_components(working, graph, repaired.label)
    for label_id, islands in groups.items():
        assert len(islands) == 1, f"label {label_id} still has {len(islands)} disconnected islands"


def test_7_3_atomic_dump_logic():
    """7.3 -- an orphaned island is donated to the bordering region with the
    smallest current area.

    Hand-built rather than relying on natural growth to produce a split: a
    "home" patch (3 faces) and a lone face on the far side of the sphere are
    both labelled 1, so label 1 has two disconnected islands. The lone face's
    three neighbors are split so exactly one neighboring label has a tiny
    area and the rest of the sphere (label 0) has a huge one -- an
    unambiguous smallest-area target.
    """
    working, _ = ingest(fixture_a_sphere())
    graph = build_dual_graph(working)
    face_count = working.face_count

    home_patch = [0] + [n for n, _c, _b in graph.adjacency[0]][:2]
    centroid0 = working.face_centroids[0]
    far_face = int(np.argmax(np.linalg.norm(working.face_centroids - centroid0, axis=1)))
    far_neighbors = [n for n, _c, _b in graph.adjacency[far_face]]
    assert far_face not in home_patch and set(far_neighbors).isdisjoint(home_patch)

    label = np.zeros(face_count, dtype=np.int64)  # label 0: rest of the sphere
    label[home_patch] = 1
    label[far_face] = 1
    tiny_label = 2
    label[far_neighbors[0]] = tiny_label  # label 2: a single tiny-area face

    region_area = np.array([float(working.face_areas[label == r].sum()) for r in range(3)])
    assert region_area[tiny_label] < region_area[0]

    new_label, new_area, dumped = apply_dump(working, graph, label, n_regions=3)

    assert len(dumped) == 1
    orphan_face, from_label, to_label = dumped[0]
    assert orphan_face == far_face
    assert from_label == 1
    assert to_label == tiny_label
    assert new_label[far_face] == tiny_label
    assert new_area[tiny_label] > region_area[tiny_label]


def test_7_4_eigenvalue_elongation():
    """7.4 -- a long, straight region exceeds the elongation threshold and
    has its penalty exponent raised for the next pass.
    """
    _, _, _, _, repaired = _run_pipeline(fixture_e_ribbon, n_pieces=2)

    assert np.any(repaired.elongation > DEFAULT_GAMMA)
    elongated = np.nonzero(repaired.elongation > DEFAULT_GAMMA)[0]
    for region_id in elongated:
        assert repaired.next_alpha[region_id] > DEFAULT_ALPHA

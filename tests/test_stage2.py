"""Stage 2 validation, per docs/mesh_partitioning_test_spec_v3.md."""

from __future__ import annotations

import numpy as np

from meshpartition.mesh import RawMesh
from meshpartition.stage0 import ingest
from meshpartition.stage1 import triage
from meshpartition.stage2 import build_bridges

from fixtures import fixture_c_shirt, fixture_d_gem


def test_2_1_gap_cap_limit():
    """2.1 -- a button moved 10m away finds no bridge and is promoted to Major."""
    raw = fixture_c_shirt()
    working, _ = ingest(raw)
    result = triage(working, n_pieces=5)

    # Relocate one button (a satellite) 10 meters away, then re-run Stage 0/1
    # on the mutated positions so the far button is still its own component.
    far_positions = working.positions.copy()
    far_satellite = result.satellite_ids[0]
    far_faces = np.unique(working.faces_pos[result.face_component_id == far_satellite])
    far_positions[far_faces] += np.array([10.0, 0.0, 0.0])

    moved_raw = RawMesh(positions=far_positions, faces_pos=working.faces_pos)
    moved_working, _ = ingest(moved_raw)
    moved_result = triage(moved_working, n_pieces=5)

    bridges = build_bridges(moved_working, moved_result)

    assert far_satellite in bridges.promoted_ids
    assert far_satellite not in bridges.satellite_ids
    assert far_satellite in bridges.major_ids


def test_2_2_normal_agreement():
    """2.2 -- a satellite anchors to the panel its own normal agrees with,
    even when a different panel's surface is physically closer.

    Fixture C's front/back panels are ~180 deg apart, so raw nearest-distance
    search never even sees the back panel as a candidate -- there's no
    ambiguity to resolve. This synthetic two-panel setup isolates the normal
    agreement mechanism itself: a small "button" sits between two opposing
    flat panels, physically nearer to the back one, but facing (and thus
    belonging to) the front one.
    """
    front_z, back_z, button_z = 0.10, -0.05, 0.0
    half = 0.5

    def flat_panel(z: float, flip: bool):
        corners = [[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]]
        tris = [[0, 1, 2], [0, 2, 3]] if not flip else [[0, 2, 1], [0, 3, 2]]
        return np.array(corners, dtype=np.float64), np.array(tris, dtype=np.int64)

    front_pos, front_faces = flat_panel(front_z, flip=False)  # normal +z
    back_pos, back_faces = flat_panel(back_z, flip=True)  # normal -z
    button_pos, button_faces = flat_panel(button_z, flip=False)  # small, normal +z
    button_pos *= 0.1  # shrink so it's clearly a small satellite, not a third panel

    positions = np.concatenate([front_pos, back_pos, button_pos], axis=0)
    faces = np.concatenate(
        [front_faces, back_faces + len(front_pos), button_faces + len(front_pos) + len(back_pos)],
        axis=0,
    )
    raw = RawMesh(positions=positions, faces_pos=faces)

    working, _ = ingest(raw)
    result = triage(working, n_pieces=3, beta=0.05)
    # Loosen the gap cap to its spec-documented maximum (c_tau=2.0, tau_max=0.10
    # x bbox diag) so this synthetic case isolates normal agreement alone --
    # the panel gap here (0.10-0.15) exists only to make "closer" concrete,
    # not to exercise the gap cap itself (that's test 2.1's job).
    bridges = build_bridges(working, result, c_tau=2.0, tau_max_fraction=0.10)

    button_component = min(
        result.satellite_ids, key=lambda c: result.component_areas[c]
    )
    front_component = max(
        result.major_ids, key=lambda c: result.component_centroids[c][2]  # highest z = front
    )

    assert bridges.anchors[button_component].other_component == front_component


def test_2_3_enclosure_override():
    """2.3 -- the gem anchors to its setting (winding number > 0.5), not the decoy floor."""
    working, _ = ingest(fixture_d_gem())
    result = triage(working, n_pieces=3, beta=0.05)
    bridges = build_bridges(working, result)

    gem_component = min(result.satellite_ids, key=lambda c: result.component_areas[c])
    setting_component = max(result.major_ids, key=lambda c: result.component_centroids[c][2])
    floor_component = min(result.major_ids, key=lambda c: result.component_centroids[c][2])
    assert setting_component != floor_component

    assert bridges.winding_numbers[(gem_component, setting_component)] > 0.5
    assert bridges.winding_numbers[(gem_component, floor_component)] <= 0.5
    assert bridges.anchors[gem_component].other_component == setting_component
    assert bridges.anchors[gem_component].via_enclosure


def test_2_4_anchor_termination():
    """2.4 -- every satellite's bridge chain terminates on a Major component."""
    working, _ = ingest(fixture_c_shirt())
    result = triage(working, n_pieces=5)
    bridges = build_bridges(working, result)

    for satellite, anchor in bridges.anchors.items():
        assert anchor.other_component in result.major_ids


def test_2_5_force_exact_count():
    """force_exact_count=True force-adopts the nearest-centroid host instead
    of promoting -- same setup as 2.1 (button moved 10m away), but the
    button now bridges (with forced=True) rather than becoming its own
    major. Default behavior (2.1) is unaffected -- force_exact_count is
    opt-in per call.
    """
    raw = fixture_c_shirt()
    working, _ = ingest(raw)
    result = triage(working, n_pieces=5)

    far_positions = working.positions.copy()
    far_satellite = result.satellite_ids[0]
    far_faces = np.unique(working.faces_pos[result.face_component_id == far_satellite])
    far_positions[far_faces] += np.array([10.0, 0.0, 0.0])

    moved_raw = RawMesh(positions=far_positions, faces_pos=working.faces_pos)
    moved_working, _ = ingest(moved_raw)
    moved_result = triage(moved_working, n_pieces=5)

    bridges = build_bridges(moved_working, moved_result, force_exact_count=True)

    assert far_satellite not in bridges.promoted_ids
    assert far_satellite in bridges.satellite_ids
    assert bridges.anchors[far_satellite].forced
    assert bridges.anchors[far_satellite].other_component == moved_result.satellite_host[far_satellite]

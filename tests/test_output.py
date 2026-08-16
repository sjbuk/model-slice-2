"""Output-contract validation, per docs/OUTPUT_FORMAT.md."""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from meshpartition.normalize import normalize_mesh
from meshpartition.output import compute_adjacency, write_puzzle_output
from meshpartition.pipeline import run_pipeline

from fixtures import fixture_a_sphere, fixture_c_shirt


def test_normalize_scale_center_ground():
    """section 1 -- longest extent is 1.0, centered on X/Z, grounded on Y."""
    raw = fixture_a_sphere()
    normalized = normalize_mesh(raw)

    mins = normalized.positions.min(axis=0)
    maxs = normalized.positions.max(axis=0)
    extents = maxs - mins

    assert abs(float(np.max(extents)) - 1.0) < 1e-9
    assert abs((mins[0] + maxs[0]) / 2.0) < 1e-9
    assert abs((mins[2] + maxs[2]) / 2.0) < 1e-9
    assert abs(float(mins[1])) < 1e-9


def _run(fixture_fn, n_pieces, out_dir):
    result = run_pipeline(fixture_fn(), n_pieces=n_pieces)
    write_puzzle_output(out_dir, result.working, result.pieces)
    return result


def test_pieces_glb_structure(tmp_path):
    """section 3 -- exactly 2 x piece_count nodes, correctly named and paired."""
    result = _run(fixture_a_sphere, n_pieces=5, out_dir=tmp_path)
    piece_count = len(result.pieces)

    scene = trimesh.load(tmp_path / "pieces.glb")
    assert len(scene.geometry) == 2 * piece_count

    names = set(scene.geometry.keys())
    for i in range(piece_count):
        assert f"piece_{i:04d}_front" in names
        assert f"piece_{i:04d}_back" in names


def test_back_mesh_shares_vertex_buffer_reversed_winding(tmp_path):
    """section 3 -- back mesh is the front mesh with faces[:, ::-1]."""
    result = _run(fixture_a_sphere, n_pieces=5, out_dir=tmp_path)
    scene = trimesh.load(tmp_path / "pieces.glb")

    front = scene.geometry["piece_0000_front"]
    back = scene.geometry["piece_0000_back"]

    assert np.allclose(front.vertices, back.vertices)
    assert np.array_equal(front.faces, back.faces[:, ::-1])


def test_back_mesh_colours_distinct(tmp_path):
    """section 3 -- each piece's back mesh samples a distinct atlas colour."""
    result = _run(fixture_a_sphere, n_pieces=5, out_dir=tmp_path)
    scene = trimesh.load(tmp_path / "pieces.glb")

    colours = []
    for i in range(len(result.pieces)):
        back = scene.geometry[f"piece_{i:04d}_back"]
        image = back.visual.material.baseColorTexture
        u, v = back.visual.uv[0]
        px = int(u * image.width) % image.width
        py = int((1.0 - v) * image.height) % image.height
        colours.append(tuple(image.getpixel((px, py))))

    assert len(set(colours)) == len(colours)


def test_checkpoint_required_fields(tmp_path):
    """section 4 -- required fields present, correctly shaped."""
    result = _run(fixture_a_sphere, n_pieces=5, out_dir=tmp_path)
    piece_count = len(result.pieces)

    payload = json.loads((tmp_path / "checkpoint.json").read_text())

    assert payload["piece_count"] == piece_count
    assert isinstance(payload["gap"], float)
    assert payload["seed"] is None
    assert len(payload["total_bounds"]["center"]) == 3
    assert len(payload["total_bounds"]["extents"]) == 3
    assert len(payload["piece_centroids"]) == piece_count
    assert all(len(c) == 3 for c in payload["piece_centroids"])
    assert len(payload["piece_vertex_counts"]) == piece_count
    assert len(payload["piece_rotation_centers"]) == piece_count
    assert isinstance(payload["adjacency"], list)


def test_checkpoint_adjacency_symmetric(tmp_path):
    """section 5 -- (a,b,offset) implies (b,a,-offset) is also present."""
    result = _run(fixture_c_shirt, n_pieces=4, out_dir=tmp_path)
    payload = json.loads((tmp_path / "checkpoint.json").read_text())

    by_pair = {(e["piece_a"], e["piece_b"]): e["offset"] for e in payload["adjacency"]}
    assert len(by_pair) == len(payload["adjacency"])  # no duplicate directed pairs

    for (a, b), offset in by_pair.items():
        assert a != b
        reverse = by_pair[(b, a)]
        assert np.allclose(offset, [-x for x in reverse])


def test_adjacency_threshold_touching_vs_separated():
    """section 5 -- AABB intersection (expanded by threshold) gates emission."""
    from meshpartition.mesh import RawMesh

    def box(cx):
        h = 0.5
        positions = np.array(
            [[cx - h, -h, -h], [cx + h, -h, -h], [cx + h, h, -h], [cx - h, h, -h]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        return RawMesh(positions=positions, faces_pos=faces)

    touching = [box(0.0), box(0.99)]  # bboxes overlap (half-extent 0.5 each)
    separated = [box(0.0), box(10.0)]

    centroids_touching = [p.positions.mean(axis=0) for p in touching]
    centroids_separated = [p.positions.mean(axis=0) for p in separated]

    adjacency_touching = compute_adjacency(touching, centroids_touching, threshold=0.01)
    adjacency_separated = compute_adjacency(separated, centroids_separated, threshold=0.01)

    assert len(adjacency_touching) == 2  # (0,1) and (1,0)
    assert len(adjacency_separated) == 0

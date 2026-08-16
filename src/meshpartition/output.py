"""Stage 10, item 3 -- write meshes and JSON manifest.

Serializes the per-piece RawMesh list produced by stage10.extract() into
the Unity-consumable output contract described in docs/OUTPUT_FORMAT.md:
a single pieces.glb (two mesh nodes per piece -- a front/cut-surface mesh
and a flat-coloured "back" lid sharing the same vertex buffer) plus a
checkpoint.json manifest carrying piece metadata and a bidirectional
bounding-box adjacency graph.
"""

from __future__ import annotations

import colorsys
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from .mesh import RawMesh, WorkingMesh
from .stage10 import region_colors
from .util import triangle_areas, triangle_centroids

ATLAS_CELL_PX = 32
ATLAS_PADDING_PX = 2
DEFAULT_ADJACENCY_THRESHOLD = 0.01
DEFAULT_GAP = 0.001


def _node_name(index: int, side: str) -> str:
    return f"piece_{index:04d}_{side}"


def build_colour_atlas(n_pieces: int) -> tuple[Image.Image, list[tuple[float, float]]]:
    """Square power-of-two colour atlas, one solid-colour cell per piece.

    Returns the image plus each piece's flat UV coordinate (cell center,
    v measured from the top per docs/OUTPUT_FORMAT.md section 3).
    """
    cols = max(1, math.ceil(math.sqrt(n_pieces)))
    rows = max(1, math.ceil(n_pieces / cols))

    cell = ATLAS_CELL_PX + ATLAS_PADDING_PX
    size = 1
    while size < max(cols, rows) * cell:
        size *= 2

    image = Image.new("RGB", (size, size), (0, 0, 0))
    pixels = image.load()

    uvs: list[tuple[float, float]] = []
    for i in range(n_pieces):
        row, col = divmod(i, cols)
        hue = (i / n_pieces) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        color = (int(r * 255), int(g * 255), int(b * 255))

        x0, y0 = col * cell, row * cell
        for y in range(y0, y0 + ATLAS_CELL_PX):
            for x in range(x0, x0 + ATLAS_CELL_PX):
                pixels[x, y] = color

        u = (col + 0.5) * cell / size
        v = 1.0 - (row + 0.5) * cell / size
        uvs.append((u, v))

    return image, uvs


def _front_mesh(piece: RawMesh) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=piece.positions, faces=piece.faces_pos, process=False)
    material = trimesh.visual.material.SimpleMaterial(diffuse=(200, 200, 200, 255))
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def _back_mesh(piece: RawMesh, atlas_image: Image.Image, uv: tuple[float, float]) -> trimesh.Trimesh:
    reversed_faces = piece.faces_pos[:, ::-1]
    mesh = trimesh.Trimesh(vertices=piece.positions, faces=reversed_faces, process=False)
    per_vertex_uv = np.tile(np.array(uv, dtype=np.float64), (len(piece.positions), 1))
    material = trimesh.visual.material.SimpleMaterial(image=atlas_image)
    mesh.visual = trimesh.visual.TextureVisuals(uv=per_vertex_uv, material=material)
    return mesh


def write_pieces_glb(out_path: Path, pieces: list[RawMesh]) -> None:
    n_pieces = len(pieces)
    atlas_image, atlas_uvs = build_colour_atlas(n_pieces)

    scene = trimesh.Scene()
    for i, piece in enumerate(pieces):
        front = _front_mesh(piece)
        back = _back_mesh(piece, atlas_image, atlas_uvs[i])
        scene.add_geometry(front, node_name=_node_name(i, "front"), geom_name=_node_name(i, "front"))
        scene.add_geometry(back, node_name=_node_name(i, "back"), geom_name=_node_name(i, "back"))

    glb_bytes = scene.export(file_type="glb", include_normals=False)
    out_path.write_bytes(glb_bytes)


def _piece_bounds(piece: RawMesh) -> tuple[np.ndarray, np.ndarray]:
    return piece.positions.min(axis=0), piece.positions.max(axis=0)


def _piece_bbox_center(piece: RawMesh) -> np.ndarray:
    mins, maxs = _piece_bounds(piece)
    return (mins + maxs) / 2.0


def _piece_mass_center(piece: RawMesh) -> np.ndarray:
    corner_positions = piece.face_corner_positions()
    areas = triangle_areas(corner_positions)
    centroids = triangle_centroids(corner_positions)
    total_area = float(np.sum(areas))
    if total_area <= 0.0:
        return piece.positions.mean(axis=0)
    return np.sum(centroids * areas[:, None], axis=0) / total_area


def compute_adjacency(
    pieces: list[RawMesh],
    centroids: list[np.ndarray],
    threshold: float = DEFAULT_ADJACENCY_THRESHOLD,
) -> list[dict]:
    """Bidirectional AABB-intersection adjacency graph, per
    docs/OUTPUT_FORMAT.md section 5.
    """
    bounds = [_piece_bounds(p) for p in pieces]
    adjacency: list[dict] = []

    for i in range(len(pieces)):
        mins_i, maxs_i = bounds[i]
        mins_i = mins_i - threshold
        maxs_i = maxs_i + threshold
        for j in range(len(pieces)):
            if i == j:
                continue
            mins_j, maxs_j = bounds[j]
            if np.all(mins_i <= maxs_j) and np.all(mins_j <= maxs_i):
                offset = centroids[i] - centroids[j]
                adjacency.append(
                    {"piece_a": i, "piece_b": j, "offset": [float(x) for x in offset]}
                )

    return adjacency


def write_checkpoint(
    out_path: Path,
    working: WorkingMesh,
    pieces: list[RawMesh],
    gap: float = DEFAULT_GAP,
    seed: int | None = None,
    source: str = "normalized.glb",
) -> None:
    mins = working.positions.min(axis=0)
    maxs = working.positions.max(axis=0)
    center = (mins + maxs) / 2.0
    extents = maxs - mins

    piece_centroids = [_piece_bbox_center(p) for p in pieces]
    piece_rotation_centers = [_piece_mass_center(p) for p in pieces]
    adjacency = compute_adjacency(pieces, piece_centroids)

    payload = {
        "piece_count": len(pieces),
        "gap": gap,
        "seed": seed,
        "total_bounds": {
            "center": [float(x) for x in center],
            "extents": [float(x) for x in extents],
        },
        "piece_centroids": [[float(x) for x in c] for c in piece_centroids],
        "adjacency": adjacency,
        "source": source,
        "piece_vertex_counts": [len(p.positions) for p in pieces],
        "piece_rotation_centers": [[float(x) for x in c] for c in piece_rotation_centers],
        "lowpoly_vertices": None,
        "lowpoly_faces": None,
        "smooth_edges": False,
        "smooth_iterations": 1,
        "smooth_lambda": 0.5,
        "smooth_nu": 0.5,
    }

    out_path.write_text(json.dumps(payload, indent=2))


def write_lowpoly_preview(
    out_path: Path,
    pieces: list[RawMesh],
    target_faces: int = 2000,
    texture_image: Image.Image | None = None,
) -> tuple[int, int]:
    """Decimated single-mesh preview (docs/OUTPUT_FORMAT.md section 2 --
    UI only, not read by Unity). Returns (vertex_count, face_count).

    All pieces are merged into one mesh -- welding shared piece-boundary
    vertices together -- before decimating as a single pass, rather than
    decimating each piece independently: independent per-piece decimation
    moves each side of a shared seam on its own, tearing open gaps at
    piece boundaries that used to be exactly coincident. Quadric edge
    collapse never opens new holes in a mesh that's already
    manifold/watertight at the seam, so welding first is what keeps the
    result closed.

    Attributes don't survive decimation (trimesh's quadric decimation, a
    thin wrapper around `fast_simplification`, drops vertex-colour/UV and
    resets the result to flat default grey), so after decimating a plain
    position/face mesh, each surviving vertex's colour or UV is pulled
    from its nearest pre-decimation *face corner* -- not the nearest
    pre-decimation *vertex*: a UV atlas seam puts multiple valid UVs at
    the same welded position (one per island touching there), and
    collapsing those onto a single vertex-indexed UV before the nearest-
    neighbour lookup would silently keep only the last one written,
    smearing every seam's texture across the wrong island. Matching by
    corner instead means every candidate UV stays available, and the
    lookup picks whichever corner is geometrically closest.

    When `texture_image` is given and every piece carries source UVs, the
    preview is textured with the model's original material instead of the
    flat per-piece colours the "Combined" webapp viewer uses.
    """
    from .stage10 import region_colors

    n_pieces = len(pieces)
    if n_pieces == 0:
        out_path.write_bytes(trimesh.Trimesh().export(file_type="glb"))
        return 0, 0

    use_texture = texture_image is not None and all(
        p.uvs is not None and p.faces_uv is not None for p in pieces
    )
    colours = region_colors(n_pieces)

    positions_parts, faces_parts, colour_parts = [], [], []
    corner_position_parts, corner_uv_parts = [], []
    offset = 0
    for i, piece in enumerate(pieces):
        positions_parts.append(piece.positions)
        faces_parts.append(piece.faces_pos + offset)
        if use_texture:
            corner_position_parts.append(piece.positions[piece.faces_pos.reshape(-1)])
            corner_uv_parts.append(piece.uvs[piece.faces_uv.reshape(-1)])

        r, g, b = colours[i]
        rgba = np.array([[int(r * 255), int(g * 255), int(b * 255), 255]], dtype=np.uint8)
        colour_parts.append(np.repeat(rgba, len(piece.positions), axis=0))
        offset += len(piece.positions)

    positions = np.concatenate(positions_parts, axis=0)
    faces = np.concatenate(faces_parts, axis=0)
    source_colours = np.concatenate(colour_parts, axis=0)

    merged = trimesh.Trimesh(vertices=positions, faces=faces, process=True)
    if len(merged.faces) > target_faces:
        merged = merged.simplify_quadric_decimation(face_count=max(4, target_faces))

    if use_texture:
        corner_positions = np.concatenate(corner_position_parts, axis=0)
        corner_uvs = np.concatenate(corner_uv_parts, axis=0)
        _, nearest_corner = cKDTree(corner_positions).query(merged.vertices, k=1)
        material = trimesh.visual.material.SimpleMaterial(image=texture_image)
        merged.visual = trimesh.visual.TextureVisuals(uv=corner_uvs[nearest_corner], material=material)
    else:
        _, nearest_vertex = cKDTree(positions).query(merged.vertices, k=1)
        merged.visual = trimesh.visual.ColorVisuals(mesh=merged, vertex_colors=source_colours[nearest_vertex])

    out_path.write_bytes(merged.export(file_type="glb"))
    return len(merged.vertices), len(merged.faces)


def write_puzzle_output(
    out_dir: Path,
    working: WorkingMesh,
    pieces: list[RawMesh],
    gap: float = DEFAULT_GAP,
    seed: int | None = None,
    source: str = "normalized.glb",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pieces_glb(out_dir / "pieces.glb", pieces)
    write_checkpoint(out_dir / "checkpoint.json", working, pieces, gap=gap, seed=seed, source=source)

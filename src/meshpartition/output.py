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
import tempfile
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


def _simplify_pymeshlab(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Decimate via PyMeshLab quadric edge collapse.

    ``preserveboundary=True`` (the default) blocks boundary-edge collapses
    that would otherwise open holes -- unlike trimesh's
    ``simplify_quadric_decimation`` (a thin wrapper around
    ``fast_simplification``), which has no such safeguard and was tearing
    open welded piece-seam vertices under heavy decimation.
    """
    import os as _os

    # pymeshlab bundles Qt5, which tries to open a display at import time
    # on Linux; the offscreen platform avoids that.
    if "QT_QPA_PLATFORM" not in _os.environ:
        _os.environ["QT_QPA_PLATFORM"] = "offscreen"

    import pymeshlab

    verts_in = np.asarray(mesh.vertices, dtype=np.float64)
    faces_in = np.asarray(mesh.faces, dtype=np.int32)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=verts_in, face_matrix=faces_in), "input")

    ms.meshing_remove_duplicate_vertices()
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        optimalplacement=True,
        preservenormal=True,
        preservetopology=True,
        qualitythr=0.5,
    )
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()

    out = ms.current_mesh()
    verts_out = np.asarray(out.vertex_matrix(), dtype=np.float64)
    faces_out = np.asarray(out.face_matrix(), dtype=np.int32)

    return trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=False)


def _simplify_pymeshlab_textured(
    positions: np.ndarray,
    faces: np.ndarray,
    corner_uvs: np.ndarray,
    texture_image: Image.Image,
    target_faces: int,
) -> trimesh.Trimesh:
    """Texture-aware quadric decimation via PyMeshLab.

    Decimating on geometry alone and reconstructing UV afterward (as
    ``_simplify_pymeshlab`` does for the flat-colour path) breaks down for
    models with multiple UV islands that sit close together in 3D -- e.g. a
    helmet visor a few millimetres from the face, or straps against a shell.
    A post-hoc nearest-position lookup can't tell which island a decimated
    vertex belongs to and silently grabs whichever original corner happens
    to be spatially closest, scrambling the texture at every such seam.

    ``meshing_decimation_quadric_edge_collapse_with_texture`` avoids this by
    folding UV distortion into the quadric error metric during collapse
    itself, so wedge (per-face-corner) UVs stay coherent through
    simplification instead of being reconstructed blind afterward.

    PyMeshLab's raw ``pymeshlab.Mesh(..., w_tex_coords_matrix=...)``
    constructor doesn't wire wedge UVs to a texture consistently -- the
    filter then rejects the mesh ("some faces without texture") -- so the
    merged mesh is round-tripped through a temporary OBJ/MTL instead, which
    is the path PyMeshLab's own OBJ importer sets up correctly.
    """
    if len(faces) > target_faces:
        import os as _os

        if "QT_QPA_PLATFORM" not in _os.environ:
            _os.environ["QT_QPA_PLATFORM"] = "offscreen"

        import pymeshlab

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tex_path = tmp_path / "texture.png"
            texture_image.convert("RGB").save(tex_path)

            obj_path = tmp_path / "merged.obj"
            mtl_path = tmp_path / "merged.mtl"
            with open(obj_path, "w") as f:
                f.write(f"mtllib {mtl_path.name}\nusemtl m\n")
                for x, y, z in positions:
                    f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
                for u, v in corner_uvs:
                    f.write(f"vt {u:.9g} {v:.9g}\n")
                for i, (a, b, c) in enumerate(faces):
                    t0, t1, t2 = 3 * i + 1, 3 * i + 2, 3 * i + 3
                    f.write(f"f {a + 1}/{t0} {b + 1}/{t1} {c + 1}/{t2}\n")
            mtl_path.write_text(f"newmtl m\nmap_Kd {tex_path.name}\n")

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(obj_path))
            ms.meshing_decimation_quadric_edge_collapse_with_texture(
                targetfacenum=target_faces,
                preserveboundary=True,
                optimalplacement=True,
                preservenormal=True,
                qualitythr=0.5,
            )
            out = ms.current_mesh()
            positions = np.asarray(out.vertex_matrix(), dtype=np.float64)
            faces = np.asarray(out.face_matrix(), dtype=np.int32)
            corner_uvs = np.asarray(out.wedge_tex_coord_matrix(), dtype=np.float64)

    # Wedge UVs are inherently per-face-corner; explode to one vertex per
    # corner since trimesh's TextureVisuals expects per-vertex UV.
    corner_positions = positions[faces.reshape(-1)]
    mesh = trimesh.Trimesh(
        vertices=corner_positions,
        faces=np.arange(len(corner_positions)).reshape(-1, 3),
        process=False,
    )
    material = trimesh.visual.material.SimpleMaterial(image=texture_image)
    mesh.visual = trimesh.visual.TextureVisuals(uv=corner_uvs, material=material)
    return mesh


def write_lowpoly_preview(
    out_path: Path,
    pieces: list[RawMesh],
    target_faces: int = 6000,
    texture_image: Image.Image | None = None,
    source_mesh: RawMesh | WorkingMesh | None = None,
) -> tuple[int, int]:
    """Decimated single-mesh preview (docs/OUTPUT_FORMAT.md section 2 --
    UI only, not read by Unity). Returns (vertex_count, face_count).

    When `texture_image` is given and `source_mesh` carries UVs, the
    preview is decimated straight from `source_mesh` -- the pre-slice,
    post-ingest mesh, still one continuous surface per component -- via
    ``_simplify_pymeshlab_textured``, which folds UV distortion into the
    collapse metric itself. Puzzle pieces are deliberately *not* used here:
    assembled pieces reconstruct the exact same geometry as `source_mesh`
    (that's the slicer's whole invariant -- no gaps, no overlaps), so
    merging pieces back together would only add puzzle-cut seams and the
    artificial bridge geometry connecting originally-disconnected
    components (a visor, straps, buckles) for no benefit, while making the
    UV harder to keep coherent: a plain position-based weld followed by
    nearest-neighbour UV reconstruction breaks down whenever the model has
    UV islands that sit close together in 3D (e.g. a visor a few
    millimetres from the face), silently scrambling the texture there.

    Otherwise the preview uses flat per-piece colours (the "Combined"
    webapp viewer's convention), built from `pieces`: all pieces are
    merged into one mesh -- welded by position, safe here since colour,
    unlike UV, is uniform within a piece and doesn't smear across islands
    the same way -- before decimating as a single pass with
    ``_simplify_pymeshlab`` (independent per-piece decimation would move
    each side of a shared seam on its own, tearing open gaps at piece
    boundaries that used to be exactly coincident), and each surviving
    vertex takes its nearest pre-decimation vertex's colour.
    """
    from .stage10 import region_colors

    n_pieces = len(pieces)
    if n_pieces == 0:
        out_path.write_bytes(trimesh.Trimesh().export(file_type="glb"))
        return 0, 0

    use_texture = (
        texture_image is not None
        and source_mesh is not None
        and source_mesh.uvs is not None
        and source_mesh.faces_uv is not None
    )

    if use_texture:
        corner_uvs = source_mesh.uvs[source_mesh.faces_uv.reshape(-1)]
        merged = _simplify_pymeshlab_textured(
            source_mesh.positions, source_mesh.faces_pos, corner_uvs, texture_image, target_faces
        )
    else:
        positions_parts, faces_parts, colour_parts = [], [], []
        colours = region_colors(n_pieces)
        offset = 0
        for i, piece in enumerate(pieces):
            positions_parts.append(piece.positions)
            faces_parts.append(piece.faces_pos + offset)
            r, g, b = colours[i]
            rgba = np.array([[int(r * 255), int(g * 255), int(b * 255), 255]], dtype=np.uint8)
            colour_parts.append(np.repeat(rgba, len(piece.positions), axis=0))
            offset += len(piece.positions)

        positions = np.concatenate(positions_parts, axis=0)
        faces = np.concatenate(faces_parts, axis=0)
        source_colours = np.concatenate(colour_parts, axis=0)

        merged = trimesh.Trimesh(vertices=positions, faces=faces, process=True)
        if len(merged.faces) > target_faces:
            merged = _simplify_pymeshlab(merged, max(4, target_faces))

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

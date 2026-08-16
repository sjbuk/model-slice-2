"""Stage 10 -- Extraction (partial).

Gathers faces per label into compact per-piece vertex buffers, preserving
corner attributes verbatim (spec Stage 10, items 1-2). The JSON manifest
called for in item 3 isn't implemented -- this only exists so the labeled
regions from Stage 8 can be written to disk and viewed.
"""

from __future__ import annotations

import colorsys

import numpy as np

from .mesh import RawMesh, WorkingMesh


def extract(working: WorkingMesh, label: np.ndarray) -> list[RawMesh]:
    """One RawMesh per region id, with compacted position/UV/normal buffers."""
    n_regions = int(label.max()) + 1
    pieces: list[RawMesh] = []

    for region_id in range(n_regions):
        face_ids = np.nonzero(label == region_id)[0]
        if len(face_ids) == 0:
            continue

        unique_pos, faces_pos = np.unique(working.faces_pos[face_ids], return_inverse=True)
        faces_pos = faces_pos.reshape(-1, 3)
        positions = working.positions[unique_pos]

        if working.uvs is not None:
            unique_uv, faces_uv = np.unique(working.faces_uv[face_ids], return_inverse=True)
            faces_uv = faces_uv.reshape(-1, 3)
            uvs = working.uvs[unique_uv]
        else:
            faces_uv = None
            uvs = None

        if working.normals is not None:
            unique_n, faces_normal = np.unique(working.faces_normal[face_ids], return_inverse=True)
            faces_normal = faces_normal.reshape(-1, 3)
            normals = working.normals[unique_n]
        else:
            faces_normal = None
            normals = None

        pieces.append(
            RawMesh(
                positions=positions,
                faces_pos=faces_pos.astype(np.int64),
                uvs=uvs,
                faces_uv=faces_uv.astype(np.int64) if faces_uv is not None else None,
                normals=normals,
                faces_normal=faces_normal.astype(np.int64) if faces_normal is not None else None,
                material_ids=working.material_ids[face_ids],
            )
        )

    return pieces


GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def region_colors(n_regions: int) -> list[tuple[float, float, float]]:
    """Per-piece color, golden-ratio-spaced hue in HSL.

    Must match the webapp viewer's client-side `hsvColor()` (same hue
    formula, same HSL model, same saturation/lightness) so a piece's color
    in the "Combined" preview matches its color everywhere else the viewer
    shows it (piece list swatches, All pieces, Exploded, Test construction).
    """
    return [colorsys.hls_to_rgb((i * GOLDEN_RATIO_CONJUGATE) % 1.0, 0.55, 0.65) for i in range(n_regions)]


def write_preview_obj(path_obj: str, path_mtl: str, working: WorkingMesh, label: np.ndarray) -> None:
    """Single combined OBJ (sharing the working mesh's global vertex buffers)
    with one material per region, so all pieces are visible -- and
    distinguishable by color -- in one file/one viewer window.
    """
    n_regions = int(label.max()) + 1
    colors = region_colors(n_regions)
    mtl_name = path_mtl.split("/")[-1].split("\\")[-1]

    with open(path_mtl, "w", encoding="utf-8") as f:
        for i, (r, g, b) in enumerate(colors):
            f.write(f"newmtl piece_{i}\n")
            f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            f.write("Ka 0 0 0\n")
            f.write("d 1.0\n\n")

    faces_by_region: dict[int, list[int]] = {i: [] for i in range(n_regions)}
    for face_id, region_id in enumerate(label):
        faces_by_region[int(region_id)].append(face_id)

    has_uv = working.uvs is not None

    with open(path_obj, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        for x, y, z in working.positions:
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        if has_uv:
            for u, v in working.uvs:
                f.write(f"vt {u:.9g} {v:.9g}\n")

        for region_id in range(n_regions):
            face_ids = faces_by_region[region_id]
            if not face_ids:
                continue
            f.write(f"o piece_{region_id}\n")
            f.write(f"usemtl piece_{region_id}\n")
            for face_id in face_ids:
                a, b, c = working.faces_pos[face_id] + 1
                if has_uv:
                    ta, tb, tc = working.faces_uv[face_id] + 1
                    f.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")
                else:
                    f.write(f"f {a} {b} {c}\n")

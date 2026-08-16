"""Pre-slicing normalization, per docs/OUTPUT_FORMAT.md section 1.

Runs once, before Stage 0 ingest, so every downstream stage/threshold and
every coordinate in the final output (piece geometry, centroids, bounds,
adjacency offsets) already lives in this normalized space.
"""

from __future__ import annotations

import numpy as np

from .mesh import RawMesh


def normalize_mesh(raw: RawMesh) -> RawMesh:
    """Uniform-scale so the longest bbox extent is 1.0, center on X/Z, ground on Y."""
    positions = raw.positions

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    extents = maxs - mins
    scale = 1.0 / float(np.max(extents))

    scaled = positions * scale
    mins = scaled.min(axis=0)
    maxs = scaled.max(axis=0)

    offset = np.array(
        [(mins[0] + maxs[0]) / 2.0, mins[1], (mins[2] + maxs[2]) / 2.0],
        dtype=np.float64,
    )
    normalized_positions = scaled - offset

    return RawMesh(
        positions=normalized_positions,
        faces_pos=raw.faces_pos,
        uvs=raw.uvs,
        faces_uv=raw.faces_uv,
        normals=raw.normals,
        faces_normal=raw.faces_normal,
        material_ids=raw.material_ids,
    )

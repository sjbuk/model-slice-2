"""Mesh data model.

Follows the OBJ face-varying convention: position, UV, and normal are
separate attribute arrays, and each face corner indexes into each array
independently. This is what lets Stage 0 weld positions without ever
touching UV/normal data (spec A4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NO_ATTR = -1


@dataclass
class RawMesh:
    """Mesh as loaded, before any conditioning."""

    positions: np.ndarray  # (P, 3) float64
    faces_pos: np.ndarray  # (F, 3) int64, indices into positions

    uvs: np.ndarray | None = None  # (T, 2) float64
    faces_uv: np.ndarray | None = None  # (F, 3) int64, or None if no UVs

    normals: np.ndarray | None = None  # (Nn, 3) float64
    faces_normal: np.ndarray | None = None  # (F, 3) int64, or None if no normals

    material_ids: np.ndarray | None = None  # (F,) int64

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.faces_pos = np.asarray(self.faces_pos, dtype=np.int64)
        n_faces = len(self.faces_pos)

        if self.uvs is not None:
            self.uvs = np.asarray(self.uvs, dtype=np.float64)
        if self.faces_uv is not None:
            self.faces_uv = np.asarray(self.faces_uv, dtype=np.int64)

        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64)
        if self.faces_normal is not None:
            self.faces_normal = np.asarray(self.faces_normal, dtype=np.int64)

        if self.material_ids is None:
            self.material_ids = np.zeros(n_faces, dtype=np.int64)
        else:
            self.material_ids = np.asarray(self.material_ids, dtype=np.int64)

    @property
    def face_count(self) -> int:
        return len(self.faces_pos)

    def face_corner_positions(self) -> np.ndarray:
        """(F, 3, 3) array of triangle vertex positions per face."""
        return self.positions[self.faces_pos]


@dataclass
class WorkingMesh:
    """The conditioned mesh Stage 1+ operates on.

    Positions are welded; UVs/normals are passed through byte-for-byte
    from the source and remain indexed per corner.
    """

    positions: np.ndarray  # (P', 3) float64, welded
    faces_pos: np.ndarray  # (F', 3) int64, indices into welded positions

    uvs: np.ndarray | None
    faces_uv: np.ndarray | None

    normals: np.ndarray | None
    faces_normal: np.ndarray | None

    material_ids: np.ndarray  # (F',)

    face_areas: np.ndarray  # (F',) float64
    face_centroids: np.ndarray  # (F', 3) float64

    # edge key (min_vertex, max_vertex) -> list of incident face ids.
    edge_faces: dict[tuple[int, int], list[int]]
    boundary_edges: frozenset[tuple[int, int]]
    nonmanifold_edges: frozenset[tuple[int, int]]

    # Provenance: welded vertex id -> sorted list of original RawMesh
    # position indices merged into it.
    vertex_source_groups: list[list[int]]
    # Working face id -> original RawMesh face index.
    source_face_indices: np.ndarray

    @property
    def face_count(self) -> int:
        return len(self.faces_pos)

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    def total_area(self) -> float:
        return float(np.sum(self.face_areas))

    def face_corner_positions(self) -> np.ndarray:
        return self.positions[self.faces_pos]

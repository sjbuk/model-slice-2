"""Stage 0, step 1: load geometry, UVs, normals, and materials.

A minimal, dependency-free OBJ reader. Unlike most OBJ loaders (including
trimesh's default path), this does NOT merge position/UV/normal into a
single per-vertex tuple -- each attribute stays in its own array, indexed
per face corner, exactly as OBJ's v/vt/vn indexing already expresses.
That separation is what Stage 0's weld step depends on (A4).
"""

from __future__ import annotations

import numpy as np

from .mesh import RawMesh


def load_obj(path: str) -> RawMesh:
    positions: list[list[float]] = []
    uvs: list[list[float]] = []
    normals: list[list[float]] = []

    faces_pos: list[list[int]] = []
    faces_uv: list[list[int]] = []
    faces_normal: list[list[int]] = []
    material_ids: list[int] = []

    material_name_to_id: dict[str, int] = {}
    current_material = -1
    has_uv = False
    has_normal = False

    def resolve(index: int, count: int) -> int:
        # OBJ indices are 1-based; negative indices count back from the end.
        return index - 1 if index > 0 else count + index

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]

            if tag == "v":
                positions.append([float(x) for x in parts[1:4]])
            elif tag == "vt":
                uvs.append([float(x) for x in parts[1:3]])
            elif tag == "vn":
                normals.append([float(x) for x in parts[1:4]])
            elif tag == "usemtl":
                name = parts[1]
                if name not in material_name_to_id:
                    material_name_to_id[name] = len(material_name_to_id)
                current_material = material_name_to_id[name]
            elif tag == "f":
                corners = parts[1:]
                if len(corners) != 3:
                    raise ValueError(f"Only triangulated faces are supported, got: {line}")
                fp, fu, fn = [], [], []
                for corner in corners:
                    fields = corner.split("/")
                    vi = resolve(int(fields[0]), len(positions))
                    fp.append(vi)
                    if len(fields) >= 2 and fields[1] != "":
                        fu.append(resolve(int(fields[1]), len(uvs)))
                        has_uv = True
                    else:
                        fu.append(-1)
                    if len(fields) >= 3 and fields[2] != "":
                        fn.append(resolve(int(fields[2]), len(normals)))
                        has_normal = True
                    else:
                        fn.append(-1)
                faces_pos.append(fp)
                faces_uv.append(fu)
                faces_normal.append(fn)
                material_ids.append(current_material if current_material >= 0 else 0)

    return RawMesh(
        positions=np.array(positions, dtype=np.float64),
        faces_pos=np.array(faces_pos, dtype=np.int64),
        uvs=np.array(uvs, dtype=np.float64) if has_uv else None,
        faces_uv=np.array(faces_uv, dtype=np.int64) if has_uv else None,
        normals=np.array(normals, dtype=np.float64) if has_normal else None,
        faces_normal=np.array(faces_normal, dtype=np.int64) if has_normal else None,
        material_ids=np.array(material_ids, dtype=np.int64),
    )


def save_obj(path: str, mesh: RawMesh) -> None:
    """Write a RawMesh back out as OBJ, mirroring load_obj's independent
    per-corner v/vt/vn indexing. No FBX writer exists in this stack (see
    Section 8's Python prototype library list) -- OBJ is the dependency-free
    format both this loader and every downstream viewer (Blender, MeshLab,
    Unity's importer) already understand.
    """
    has_uv = mesh.uvs is not None
    has_normal = mesh.normals is not None

    with open(path, "w", encoding="utf-8") as f:
        for x, y, z in mesh.positions:
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        if has_uv:
            for u, v in mesh.uvs:
                f.write(f"vt {u:.9g} {v:.9g}\n")
        if has_normal:
            for x, y, z in mesh.normals:
                f.write(f"vn {x:.9g} {y:.9g} {z:.9g}\n")

        for face_id in range(mesh.face_count):
            corners = []
            for corner in range(3):
                vi = mesh.faces_pos[face_id][corner] + 1
                if has_uv and has_normal:
                    ti = mesh.faces_uv[face_id][corner] + 1
                    ni = mesh.faces_normal[face_id][corner] + 1
                    corners.append(f"{vi}/{ti}/{ni}")
                elif has_uv:
                    ti = mesh.faces_uv[face_id][corner] + 1
                    corners.append(f"{vi}/{ti}")
                elif has_normal:
                    ni = mesh.faces_normal[face_id][corner] + 1
                    corners.append(f"{vi}//{ni}")
                else:
                    corners.append(f"{vi}")
            f.write(f"f {' '.join(corners)}\n")

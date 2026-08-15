"""Procedural authoring of the six fixtures required by the validation plan.

Kept deliberately small (well under the 5k triangle ceiling) and free of any
randomness, so every fixture is exactly reproducible run to run.
"""

from __future__ import annotations

import math

import numpy as np

from meshpartition.mesh import RawMesh

# ---------------------------------------------------------------------------
# Fixture A -- The Sphere: watertight, UV-mapped, roughly uniform triangulation.
# ---------------------------------------------------------------------------


def _icosahedron():
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return verts, faces


def _subdivide(verts: np.ndarray, faces: np.ndarray):
    midpoint_cache: dict[tuple[int, int], int] = {}
    verts = list(map(np.array, verts))

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key in midpoint_cache:
            return midpoint_cache[key]
        m = (verts[i] + verts[j]) / 2.0
        m = m / np.linalg.norm(m)
        idx = len(verts)
        verts.append(m)
        midpoint_cache[key] = idx
        return idx

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces.append([a, ab, ca])
        new_faces.append([b, bc, ab])
        new_faces.append([c, ca, bc])
        new_faces.append([ab, bc, ca])
    return np.array(verts, dtype=np.float64), np.array(new_faces, dtype=np.int64)


def fixture_a_sphere(subdivisions: int = 2) -> RawMesh:
    verts, faces = _icosahedron()
    for _ in range(subdivisions):
        verts, faces = _subdivide(verts, faces)

    u = 0.5 + np.arctan2(verts[:, 2], verts[:, 0]) / (2 * math.pi)
    v = 0.5 - np.arcsin(np.clip(verts[:, 1], -1.0, 1.0)) / math.pi
    uvs = np.stack([u, v], axis=1)

    return RawMesh(
        positions=verts,
        faces_pos=faces,
        uvs=uvs,
        faces_uv=faces,
        normals=verts.copy(),  # unit sphere: position == normal direction
        faces_normal=faces,
    )


# ---------------------------------------------------------------------------
# Fixture B -- The Soup: open annulus with a T-junction and a degenerate face.
# ---------------------------------------------------------------------------


def fixture_b_soup(n: int = 8, outer_r: float = 1.0, inner_r: float = 0.4) -> RawMesh:
    angles = 2 * math.pi * np.arange(n) / n
    outer = np.stack([outer_r * np.cos(angles), outer_r * np.sin(angles), np.zeros(n)], axis=1)
    inner = np.stack([inner_r * np.cos(angles), inner_r * np.sin(angles), np.zeros(n)], axis=1)
    positions = np.concatenate([outer, inner], axis=0)  # outer: 0..n-1, inner: n..2n-1

    faces = []
    for i in range(n):
        i2 = (i + 1) % n
        faces.append([i, i2, n + i])  # outer_i, outer_i+1, inner_i
        faces.append([n + i, i2, n + i2])  # inner_i, outer_i+1, inner_i+1
    faces = [list(f) for f in faces]

    # T-junction: edge (outer0, inner0) is already shared by two annulus
    # triangles (i=0's first tri and i=n-1's second tri). Add a third face
    # on that same edge so it has 3 incident faces.
    flap_apex = np.array([[(outer[0, 0] + inner[0, 0]) / 2, (outer[0, 1] + inner[0, 1]) / 2, 0.1]])
    positions = np.concatenate([positions, flap_apex], axis=0)
    flap_apex_idx = len(positions) - 1
    faces.append([0, n, flap_apex_idx])  # outer0, inner0, apex

    # Degenerate (zero-area) face: reuse a single vertex twice.
    faces.append([0, 0, 1])

    faces_pos = np.array(faces, dtype=np.int64)
    return RawMesh(positions=positions, faces_pos=faces_pos)


# ---------------------------------------------------------------------------
# Fixture C -- The Shirt: a torso with three disconnected buttons.
# ---------------------------------------------------------------------------


def _box(cx: float, cy: float, cz: float, hx: float, hy: float, hz: float):
    """Axis-aligned box centered at (cx,cy,cz), returns (positions, tri faces)."""
    signs = np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        dtype=np.float64,
    )
    positions = signs * np.array([hx, hy, hz]) + np.array([cx, cy, cz])
    quads = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    faces = []
    for a, b, c, d in quads:
        faces.append([a, b, c])
        faces.append([a, c, d])
    return positions, np.array(faces, dtype=np.int64)


def fixture_c_shirt() -> RawMesh:
    """A single torso box with three disconnected satellite buttons floating
    just in front of it (one major component, three satellites).
    """
    torso_pos, torso_faces = _box(0, 0, 0, 1.0, 1.5, 0.3)

    positions = [torso_pos]
    faces = [torso_faces]
    vertex_offset = len(torso_pos)

    button_centers = [(-0.5, 0.5), (0.0, 0.0), (0.5, -0.5)]
    for bx, by in button_centers:
        bpos, bfaces = _box(bx, by, 0.35, 0.05, 0.05, 0.02)
        faces.append(bfaces + vertex_offset)
        positions.append(bpos)
        vertex_offset += len(bpos)

    return RawMesh(
        positions=np.concatenate(positions, axis=0),
        faces_pos=np.concatenate(faces, axis=0),
    )


def fixture_c_shirt_two_torsos() -> RawMesh:
    """Two disconnected major-sized torso boxes, no buttons.

    Not one of the six official fixtures -- used only by Stage 1's
    N-less-than-major-component-count infeasibility test. Plain
    fixture_c_shirt() has a single torso, which is what Stage 2's bridging
    and Stage 4's whole-graph connectivity tests need (there is no
    major-to-major bridging in this pipeline, only satellite-to-major, so a
    fixture with two genuinely disconnected majors and no shared satellite
    between them can never appear as a single connected component).
    """
    torso1_pos, torso1_faces = _box(0, 0, 0, 1.0, 1.5, 0.3)
    torso2_pos, torso2_faces = _box(10, 0, 0, 1.0, 1.5, 0.3)
    return RawMesh(
        positions=np.concatenate([torso1_pos, torso2_pos], axis=0),
        faces_pos=np.concatenate([torso1_faces, torso2_faces + len(torso1_pos)], axis=0),
    )


# ---------------------------------------------------------------------------
# Fixture D -- The Gem: a ring with a disconnected gem nested inside.
# ---------------------------------------------------------------------------


def _octahedron(center: tuple[float, float, float], r: float):
    cx, cy, cz = center
    positions = np.array(
        [
            [cx + r, cy, cz], [cx - r, cy, cz],
            [cx, cy + r, cz], [cx, cy - r, cz],
            [cx, cy, cz + r], [cx, cy, cz - r],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ],
        dtype=np.int64,
    )
    return positions, faces


def fixture_d_gem(
    n: int = 12, radius: float = 0.5, height: float = 0.6, gem_r: float = 0.08
) -> RawMesh:
    """A cup-shaped setting (side wall + floor, open top) with a gem nested
    inside it, plus a decoy floor plane well below everything.

    The setting genuinely wraps around the gem (unlike a flat ring), so its
    generalized winding number as seen from the gem is a meaningful >0.5
    "enclosed" signal -- exercising Stage 2's enclosure override against the
    decoy floor, which a naive nearest-surface search could otherwise prefer.
    """
    angles = 2 * math.pi * np.arange(n) / n
    rim = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, height)], axis=1)
    base_rim = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)], axis=1)
    bottom_center = np.array([[0.0, 0.0, 0.0]])

    setting_positions = np.concatenate([base_rim, rim, bottom_center], axis=0)
    center_idx = 2 * n

    setting_faces = []
    for i in range(n):
        i2 = (i + 1) % n
        # Side wall quad (base_rim[i], base_rim[i2], rim[i2], rim[i]).
        setting_faces.append([i, i2, n + i2])
        setting_faces.append([i, n + i2, n + i])
        # Bottom cap fan.
        setting_faces.append([i2, i, center_idx])
    setting_faces = np.array(setting_faces, dtype=np.int64)

    gem_positions, gem_faces = _octahedron((0.0, 0.0, height * 0.4), gem_r)

    floor_half = 1.0
    floor_z = -0.3
    floor_positions = np.array(
        [
            [-floor_half, -floor_half, floor_z], [floor_half, -floor_half, floor_z],
            [floor_half, floor_half, floor_z], [-floor_half, floor_half, floor_z],
        ],
        dtype=np.float64,
    )
    floor_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    positions = [setting_positions, gem_positions, floor_positions]
    faces = [
        setting_faces,
        gem_faces + len(setting_positions),
        floor_faces + len(setting_positions) + len(gem_positions),
    ]
    return RawMesh(
        positions=np.concatenate(positions, axis=0), faces_pos=np.concatenate(faces, axis=0)
    )


# ---------------------------------------------------------------------------
# Fixture E -- The Ribbon: a long, flat, thin rectangle.
# ---------------------------------------------------------------------------


def fixture_e_ribbon(length: float = 10.0, width: float = 0.2, segments: int = 40) -> RawMesh:
    xs = np.linspace(-length / 2, length / 2, segments + 1)
    top = np.stack([xs, np.full_like(xs, width / 2), np.zeros_like(xs)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, -width / 2), np.zeros_like(xs)], axis=1)
    positions = np.concatenate([top, bottom], axis=0)  # top: 0..segments, bottom offset

    n = segments + 1
    faces = []
    for i in range(segments):
        t0, t1 = i, i + 1
        b0, b1 = n + i, n + i + 1
        faces.append([t0, b0, t1])
        faces.append([t1, b0, b1])
    return RawMesh(positions=positions, faces_pos=np.array(faces, dtype=np.int64))


# ---------------------------------------------------------------------------
# Fixture F -- The UV Box: a cube authored as independent, unwelded faces --
# each face owns its own 4 corner positions (coincident with its neighbors'),
# its own UV island, and its own flat normal (a hard crease on every edge).
# ---------------------------------------------------------------------------


def fixture_f_uvbox(half_extent: float = 0.5) -> RawMesh:
    h = half_extent
    # Each entry: (4 corner positions CCW, face normal, material id)
    face_defs = [
        ([[h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h]], [1, 0, 0]),   # +X
        ([[-h, h, -h], [-h, -h, -h], [-h, -h, h], [-h, h, h]], [-1, 0, 0]),  # -X
        ([[-h, h, -h], [h, h, -h], [h, h, h], [-h, h, h]], [0, 1, 0]),   # +Y
        ([[-h, -h, h], [h, -h, h], [h, -h, -h], [-h, -h, -h]], [0, -1, 0]),  # -Y
        ([[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]], [0, 0, 1]),   # +Z
        ([[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]], [0, 0, -1]),  # -Z
    ]

    positions, normals, uvs = [], [], []
    faces_pos, faces_normal, faces_uv, material_ids = [], [], [], []

    for face_id, (corners, normal) in enumerate(face_defs):
        base = len(positions)
        positions.extend(corners)
        normals.append(normal)
        normal_idx = len(normals) - 1

        # This face's own UV island: a unit square offset so it never
        # overlaps any other face's island.
        island_uv = [[0, 0], [1, 0], [1, 1], [0, 1]]
        uv_base = len(uvs)
        uvs.extend([[u + 2 * face_id, v] for u, v in island_uv])

        quad = [base, base + 1, base + 2, base + 3]
        quad_uv = [uv_base, uv_base + 1, uv_base + 2, uv_base + 3]
        for tri in ([0, 1, 2], [0, 2, 3]):
            faces_pos.append([quad[i] for i in tri])
            faces_uv.append([quad_uv[i] for i in tri])
            faces_normal.append([normal_idx, normal_idx, normal_idx])
            material_ids.append(face_id)

    return RawMesh(
        positions=np.array(positions, dtype=np.float64),
        faces_pos=np.array(faces_pos, dtype=np.int64),
        uvs=np.array(uvs, dtype=np.float64),
        faces_uv=np.array(faces_uv, dtype=np.int64),
        normals=np.array(normals, dtype=np.float64),
        faces_normal=np.array(faces_normal, dtype=np.int64),
        material_ids=np.array(material_ids, dtype=np.int64),
    )


ALL_FIXTURES = {
    "A": fixture_a_sphere,
    "B": fixture_b_soup,
    "C": fixture_c_shirt,
    "D": fixture_d_gem,
    "E": fixture_e_ribbon,
    "F": fixture_f_uvbox,
}

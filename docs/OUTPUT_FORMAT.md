# Jigsaw Generator — Output Contract

This document specifies the **exact shape of the output directory** produced by this
project's jigsaw generator pipeline (`planar_main.py`), so a re-implementation in a
separate project can produce byte-for-byte compatible assets that the existing
Unity consumer (`UnityApp/Assets/_Project/Scripts/Core/PuzzleManager.cs`,
`CheckpointData.cs`, `SnapSystem.cs`) can load without modification.

Anything not described here (internal slicing algorithm, BSP strategy, mesh
processing) is an implementation detail — only the **files, fields, naming
conventions, and coordinate conventions below are the contract**.

## 1. Input normalization

Before slicing, the source GLB is normalized (`planar_phase_010.normalize_mesh`):

- Uniform scale so the **longest bounding-box extent = 1.0** (`scale = 1 / max(extents)`).
- Centered on X and Z (`centroid.x == 0`, `centroid.z == 0`).
- Grounded on Y (lowest vertex at `y == 0`).

All coordinates in the output (piece geometry, centroids, bounds, adjacency
offsets) are expressed in this normalized space, in a **right-handed, Y-up**
coordinate system (glTF convention — trimesh/glTF, not Unity's left-handed Y-up;
the Unity-side GLB importer, `GLTFast`, performs the axis conversion on load).

## 2. Output directory layout

```
<output_dir>/
├── pieces.glb            # consolidated multi-node scene — REQUIRED, primary runtime asset
├── checkpoint.json        # metadata — REQUIRED
├── pieces/
│   ├── piece_0000.glb      # per-piece front mesh (optional/debug — not read by Unity)
│   ├── piece_0000_back.glb # per-piece back mesh  (optional/debug — not read by Unity)
│   ├── piece_0001.glb
│   ├── piece_0001_back.glb
│   └── ...
├── colour_atlas.png       # back-face colour atlas texture referenced by pieces.glb
├── preview.png            # full-puzzle thumbnail render (UI only, not consumed by Unity)
└── lowpoly_preview.glb    # decimated single-mesh preview (UI only, not consumed by Unity)
```

Only `pieces.glb` and `checkpoint.json` are read by the Unity runtime. The
`pieces/` directory, `preview.png`, `colour_atlas.png` (as a standalone file —
it's still needed embedded/referenced inside `pieces.glb`) and
`lowpoly_preview.glb` exist for the authoring tool UI / debugging and are not
strictly required for gameplay, but should be produced for full compatibility
with the authoring app.

## 3. `pieces.glb` — consolidated scene

A single glTF binary (`trimesh.Scene`) containing **two mesh nodes per piece**:

| Node name | Content |
|---|---|
| `piece_{i:04d}_front` | The piece's outer/cut surface mesh (the actual puzzle-piece geometry, inheriting the source model's material/UVs/texture). |
| `piece_{i:04d}_back`  | A matching "lid" mesh closing the piece's back (inside) face. Same vertex buffer as the front piece, winding reversed (`faces[:, ::-1]`), with its own UVs mapping into `colour_atlas.png` so each piece gets a **single flat, unique colour** on its back side. |

- `i` is the **zero-based piece index**, zero-padded to 4 digits (`0000`–`9999`).
- Both nodes for a piece must exist and share the same index `i`.
- Node/geometry names use the pattern `piece_<4-digit-index>_<front|back>` —
  **the `piece_` prefix and the second underscore-delimited token being the
  numeric index is load-bearing**: the Unity loader parses the piece ID via
  `name.Split('_')[1]` (see `PuzzleManager.ParsePieceId`), so any extra
  suffix after the index (`_front` / `_back`) is fine, but the index must be
  the token immediately after the first `_`.
- Node names must be unique per piece pair (`i` must not repeat).
- Do not include any other top-level scene nodes with non-`piece_*` names —
  the importer treats any node whose name doesn't parse to a piece index as
  scene clutter and destroys it if it ends up as an empty leaf.
- Exported with `include_normals=False` (normals are recomputed/not required
  in the export — flat/none is acceptable).

### Back-face colour atlas

- Square, power-of-two PNG (`colour_atlas.png`), a grid of solid-colour cells
  (32px cells + 2px padding), one cell per piece, colours generated via
  evenly-spaced HSV hues (`hue = i/n, s=0.8, v=0.9`).
- Each back mesh gets **flat per-vertex UVs** all pointing at the center of
  its cell: `u = (col+0.5)/cols, v = 1 - (row+0.5)/rows` — i.e. this does not
  need to be literally an atlas; **any mechanism that gives each piece's back
  face a single distinct solid colour is functionally compatible**, since
  Unity only renders the textured back mesh — it does not read the atlas
  file or its layout independently.
- This atlas image must be embedded in / referenced by the GLB's material
  (baseColorTexture) for the back-face nodes — not just written as the
  standalone `colour_atlas.png` file.

## 4. `checkpoint.json` — metadata (REQUIRED, exact schema)

This is a flat JSON object. Fields marked **required** are read directly by
`CheckpointData.cs` via `JsonUtility.FromJson` (silently ignores unknown
extra fields, but **missing required fields deserialize to default values
with no error** — `0`, `null`, empty array — so omitting them breaks the game
without an exception).

```jsonc
{
  // REQUIRED — consumed by Unity
  "piece_count": 16,                  // int — number of pieces (== node-pairs in pieces.glb)
  "gap": 0.001,                       // float — micro-bevel gap used between piece boundaries
  "seed": null,                       // int | null — RNG seed used for slicing (nullable)
  "total_bounds": {
    "center":  [0.0, 0.5, 0.0],       // float[3] — bounding-box centroid of the whole (normalized) model
    "extents": [0.7578, 1.0, 0.7657]  // float[3] — bounding-box full extents (not half-extents)
  },
  "piece_centroids": [                // float[3][] — REQUIRED, length == piece_count
    [0.0361, 0.7852, 0.0222],         //   index i == piece i's centroid in assembled/normalized space.
    ...                               //   Used by Unity for wall-slot outward-direction & Y-rotation.
  ],
  "adjacency": [                      // REQUIRED — directed neighbour graph, see §5
    { "piece_a": 0, "piece_b": 3, "offset": [dx, dy, dz] },
    ...
  ],

  // present in generator output, not currently read by CheckpointData.cs
  // (safe to omit for a minimal port, but keep for forward-compatibility)
  "source": "normalized.glb",         // string — basename of the (normalized) source file
  "piece_vertex_counts": [123, 88, ...], // int[] — per-piece front-mesh vertex count, length == piece_count
  "piece_rotation_centers": [...],    // float[3][] — per-piece mesh center_mass, length == piece_count
  "lowpoly_vertices": 1998,           // int | null — vertex count of lowpoly_preview.glb
  "lowpoly_faces": 2000,              // int | null — face count of lowpoly_preview.glb
  "smooth_edges": false,              // bool — whether boundary smoothing was applied
  "smooth_iterations": 1,             // int
  "smooth_lambda": 0.5,               // float
  "smooth_nu": 0.5,                   // float

  // added later by the authoring backend (not by the generator itself),
  // but preserved by it across re-slices — see §6
  "name": "My Puzzle",                // string — user-facing puzzle title
  "orientation": {                    // object | null — saved camera pose for the puzzle's preview
    "position": [x, y, z],
    "target":   [x, y, z]
  }
}
```

Notes:

- `piece_centroids[i]` corresponds to `piece_{i:04d}_front`/`_back` in
  `pieces.glb` — **array order/index must exactly match the GLB node index**,
  they are matched purely positionally, not by name lookup back into the JSON.
- `adjacency` entries' `piece_a`/`piece_b` are also these same zero-based
  indices.
- All floats are plain JSON numbers (not scientific-notation-only — standard
  Python `json.dump` output is fine, `float.TryParse` on the C# side handles
  exponents too).
- Written with `json.dump(..., indent=2)` — pretty-printing is not required
  by the parser, just convention.

## 5. Adjacency semantics

`adjacency` is a **directed graph encoded as an undirected pair list — each
touching pair appears twice**, once in each direction:

```jsonc
{ "piece_a": i, "piece_b": j, "offset": [x, y, z] }
```

- Emitted whenever piece `i`'s and piece `j`'s axis-aligned bounding boxes
  (expanded by `adjacency_threshold`, default `0.01` in normalized units)
  intersect.
- `offset = centroid[i] - centroid[j]` — i.e. **the vector from piece j to
  piece i**, in the same normalized/assembled space as `piece_centroids`.
- Both `(i, j, offset)` and `(j, i, -offset)` are present as separate
  entries — consumers may look up either direction.
- Self-pairs (`i == j`) are never emitted.
- Unity (`SnapSystem.cs`) builds a `Dictionary<(pieceA,pieceB), offset>` from
  this list and uses it to test whether two currently-held pieces are in
  their correct relative position before snapping them together — so the
  offset's sign/direction convention above must be preserved exactly.

## 6. Fields owned by the authoring backend, not the generator

`name` and `orientation` are written/merged into `checkpoint.json` by the
Tauri backend (`tauri-app/backend/routes.py`), not by the core generator —
but the generator's re-slice operations (smoothing, orphan-fix, etc.) must
**preserve** any existing `name`/`orientation` keys already in
`checkpoint.json` when they overwrite it (read old values first, merge back
in after writing the fresh checkpoint). If your separate project's generator
is standalone (no such backend), you can omit these two fields entirely —
Unity treats both as optional/nullable.

## 7. Minimal compatibility checklist

To be a drop-in replacement for `PuzzleManager.LoadPuzzle()`:

1. Output directory contains `pieces.glb` and `checkpoint.json` at its root.
2. `pieces.glb` has exactly `2 × piece_count` nodes named
   `piece_<4-digit-index>_front` / `piece_<4-digit-index>_back`, indices
   `0 … piece_count-1`, no gaps.
3. Each `_back` node is a valid closed-ish mesh with its own solid colour
   (via UV+texture or vertex colours baked into a texture) distinct from
   every other piece's back colour.
4. `checkpoint.json` has `piece_count`, `piece_centroids` (length ==
   `piece_count`, index-aligned to the GLB piece nodes), and `adjacency`
   (bidirectional, `offset = centroid[a] - centroid[b]`).
5. Coordinates are right-handed Y-up, piece geometry pre-assembled in its
   final "solved" position (Unity does not translate pieces into place using
   `piece_centroids` — it reads centroids only for auxiliary calculations
   like outward wall-facing direction — so the mesh geometry itself must
   already be sitted in solved-puzzle space).

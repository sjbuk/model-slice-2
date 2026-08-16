# Mesh partitioning (VR jigsaw)

Offline pipeline that splits an arbitrary triangle mesh into N connected,
roughly equal-area pieces for use as a VR jigsaw puzzle. Full design intent
lives in [`docs/mesh_partitioning_spec_v3.md`](docs/mesh_partitioning_spec_v3.md);
this file covers running it and the performance characteristics of the
current implementation.

## Running it

**CLI**, against an OBJ file:

```
python scripts/run_pipeline.py <path-to-obj> [n_pieces] [--force-exact-count]
```

Writes per-piece OBJs plus a combined colored preview to `output/<model-name>/`.

**Web UI** (upload FBX/GLB/OBJ, pick a piece count, preview the result):

```
docker compose -f docker/docker-compose.yml up meshpartition-web
```

then open `http://localhost:5000`. `assimp` (bundled in the image) converts
non-OBJ uploads before slicing.

**Tests**: `docker compose -f docker/docker-compose.yml run --rm meshpartition-tests`,
or `pytest` locally (`tests/conftest.py` puts `src/` on the path).

## Performance notes

Stages 0/1/4–7 are effectively linear in face count and stay fast even on
large, messy real-world assets. Two stages don't, by construction, and both
have been specifically shaped around that:

### Stage 2 — bridge construction

Satellite-to-major bridging is fundamentally a nearest-triangle search:
brute force is `O(sum of satellite faces × total major faces)`, which is
billions of triangle-distance evaluations on a real hard-surface asset
(hundreds of disconnected components) and effectively never finishes.
`stage2.py` bounds this two ways:

- A KD-tree over major-face centroids restricts each satellite face's
  candidate set to majors that could plausibly land within its gap threshold
  (`tau_s`), sized off the 99th-percentile face extent rather than the true
  max so a handful of oversized panel faces can't blow up every query's
  radius — those rare larger faces get their own small, separately-queried
  candidate tree instead of being spliced unconditionally into every
  satellite's candidates.
- The exact closest-point-on-triangle evaluations that survive the
  prefilter run through a numba-compiled kernel (`util.closest_point_distance_matrix`)
  instead of vectorized numpy, which was allocating ~20 temporary arrays per
  call — the compiled version allocates nothing beyond the output distance
  matrix and parallelizes over query points.

Net effect on a 213k-face reference asset (hundreds of disconnected
components): Stage 2 dropped from unbounded (never completing) to
consistently well under a minute.

### Stage 8 — Lloyd relaxation / region medoids

Each relaxation pass re-seeds every region at its graph medoid — the face
minimizing total dual-graph distance to the rest of the region. Computed
exactly, that's an all-pairs Dijkstra: a dense `n × n` distance matrix per
region, `O(n²)` memory. Fine when regions are "fixture-scale," but a mesh
with few, large regions (e.g. `--force-exact-count` on an asset with lots of
disconnected detail parts and not many true major components) can leave a
single region with tens of thousands of faces — a dense matrix at that size
is gigabytes, and building it every iteration for every such region
exhausted a 31GB container before this was addressed.

`stage8.py` now computes the medoid exactly only up to `_EXACT_MEDOID_MAX_FACES`
faces. Past that, it evaluates a deterministic, evenly-spaced sample of
`_APPROX_MEDOID_SAMPLE_SIZE` candidates instead of every face — each still
scored by a genuine single-source Dijkstra (one row, `O(n)` memory, not the
whole matrix), just not exhaustively over all `n` candidates. Lloyd
relaxation only needs a reasonable re-seed point each pass, not the true
minimum, and the loop self-corrects over iterations regardless.

This fixes the memory blowup outright (verified: a run that previously hit
~30GB now holds under 1GB for the same input). It does **not** fully fix the
*speed* of that specific worst case — exhaustively relaxing a handful of
enormous regions over up to 15 iterations is still slow, just bounded. A
sane piece count for a given asset (regions that don't end up
absurdly large) avoids the worst case entirely; the web UI's `n_pieces`
choice matters here as much as any algorithmic bound.

### Web UI safety net

`webapp/app.py` kills a slicing job outright after `JOB_TIME_BUDGET_SECONDS`
(600s) rather than leaving it running (and polled) indefinitely, so a
pathological input fails visibly instead of hanging the UI forever.

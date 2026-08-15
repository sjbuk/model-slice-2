# Surface-area-balanced mesh partitioning
**Specification v3.0 — Final**

Offline authoring pipeline for splitting a 3D model into N connected pieces of approximately equal surface area, for use as a VR jigsaw. 

*Notable in v3.0: Optimized for stability over strict mathematical area balance. Area variance up to 25% is tolerated to preserve original mesh topology, protect UV boundaries, and simplify region connectivity repair.*

## 1. Purpose
Given an arbitrary triangle mesh and a target piece count N, produce N pieces such that:
*   Every piece has approximately equal surface area (±25% variance acceptable).
*   Every piece is connected — no piece is split into geometrically unrelated fragments.
*   Detached-but-associated geometry (buttons, gems, loose trim) travels with the piece it visually belongs to.
*   Seams are visually clean, respect hard UV/normal boundaries, and prefer to hide in concave creases.
*   The result is deterministic: identical input plus identical parameters yields identical output.
*   Cuts are produced by region growing over the surface, never by planar slicing. 

### 1.1 Out of scope
Gameplay, snapping behaviour, piece pivots, colliders, and runtime presentation. Physical manufacture (printability, wall thickness, magnets).

## 2. Assumptions and constraints

| # | Assumption | Consequence if violated |
| :--- | :--- | :--- |
| **A1** | Runs offline in an authoring tool. Pieces ship as baked assets. | Runtime cutting requires a native C++ core and a different library stack. |
| **A2** | Input may be non-watertight (holes, soup). | The pipeline partitions the surface as given; holes are just boundary edges. |
| **A3** | Input may contain multiple disconnected components. | Core requirement, handled in Stages 1–2. |
| **A4** | Models carry UVs, per-corner normals, and multiple materials. | **Strictly constrains** welding (Stage 0) and seam fairing (Stage 9). UVs cannot be torn. |
| **A5** | Meshes are static, not skinned. | If skinned, partition in bind pose; carry weights as an interpolated attribute. |
| **A6** | Triangle budget 10k–200k per model. | Above ~500k, requires multiresolution scheme. |
| **A7** | The cutter may introduce new vertices/re-triangulate. | Limited to necessary seam-smoothing without violating A4. |
| **A8** | N is fixed by the caller. | Handled via target piece counts. |
| **A9** | Output is deterministic. | No RNG anywhere. All ties break on stable indices. |

## 3. Definitions
*   **Working mesh:** The mesh the algorithm operates on, with a retained mapping back to the source.
*   **Cell:** One face of the working mesh. Weight = its area.
*   **Dual graph:** Nodes are cells; edges connect cells that may exchange a growth front.
*   **Bridge edge:** A dual edge between cells on different connected components, spanning a physical gap.
*   **Component:** A maximal set of faces connected by shared edges in the source mesh.
*   **Satellite:** A small component adopted into a host region.
*   **Anchor:** The single cell on another component that a satellite is bound to.
*   **Region:** A set of cells sharing a label. Becomes one output piece.
*   **$\bar{A}$ (Target Area):** The ideal area per piece ($A_{total} / N$).

## 4. Data model
*(Refer to v1.0 data model for `WorkingMesh`, `DualGraph`, `Region`, and `Manifest` schemas).*

## 5. Parameters

| Symbol | Name | Default | Range | Stage |
| :--- | :--- | :--- | :--- | :--- |
| $\delta_{weld}$ | Weld tolerance | 1e-4 × bbox diagonal | 1e-6 – 1e-3 | 0 |
| $\epsilon_{area}$ | Degenerate face area | 1e-8 × bbox diag² | — | 0 |
| $\beta$ | Major component threshold, ×$\bar{A}$ | 0.25 | 0.05 – 1.0 | 1 |
| $c_\tau$ | Bridge gap scale | 0.5 | 0.1 – 2.0 | 2 |
| $\tau_{max}$ | Absolute bridge gap cap, ×bbox diag | 0.03 | 0.005 – 0.10 | 2 |
| $m$ | Bridge candidates per satellite | 5 | 3 – 8 | 2 |
| $\theta_n$ | Normal agreement threshold | 0.0 (cos) | −0.3 – 0.5 | 2 |
| $\lambda$ | Bridge cost multiplier | 5.0 | 1.0 – 20.0 | 2 |
| $a_{max}$ | Max allowed face area, ×$\bar{A}$ | 0.25 | 0.10 – 0.50 | 3 |
| $\mu$ | Concavity weight | 2.0 | 0 – 5 | 4 |
| $\alpha$ | Capacity greed exponent | 1.5 | 0.5 – 3.0 | 6 |
| $\zeta$ | Greed-term area floor | 0.1 | 0.01 – 0.3 | 6 |
| $\gamma$ | Elongation limit (Eigenvalue ratio) | 2.5 | 1.5 – 5.0 | 7 |
| $\epsilon_{bal}$ | Area convergence target | 0.25 | 0.05 – 0.30 | 8 |
| $I_{max}$ | Max relaxation iterations | 15 | 5 – 50 | 8 |

## 6. Global invariants
**GI-1:** Total surface area is conserved from Stage 0 onward (1e-6 relative).
**GI-2:** Every cell has exactly one label, from Stage 6 onward.
**GI-3:** No NaN or infinite value in any vertex position, area, or cost.
**GI-4:** Every dual edge cost is finite, positive, and symmetric.
**GI-5:** Identical input + parameters = byte-identical output.

---

## 7. Pipeline

### Stage 0 — Ingest and conditioning
1. Load geometry, UVs, normals, materials.
2. Remove faces with area below $\epsilon_{area}$.
3. **Weld positions only**, within $\delta_{weld}$. Do not merge attributes. Vertices become shared for topology purposes while UVs and normals remain per-corner.
4. Build half-edge structure. Record boundary edges. Compute face areas and centroids.

### Stage 1 — Component triage and piece apportionment
Classify components as *major* ($A_k \ge \beta \cdot \bar{A}$) or *satellite*. Apportion the target piece count $N$ across major components based on their area plus the area of any satellites they will adopt.

### Stage 2 — Bridge construction
1. Build a BVH over all faces.
2. For each satellite, find the $m$ nearest distinct points on other components.
3. Reject hits beyond distance $\tau_S$ or where normal agreement is below $\theta_n$.
4. **Enclosure override:** Query a generalized winding number tree (built once per major component) to ensure objects like gems bridge to their settings, not the nearest external surface.
5. Emit surviving hits as bridge edges in the dual graph. Designate the best as the satellite's anchor.

### Stage 3 — Refinement (Optimized)
*Purpose:* Ensure no single triangle is so massive that it breaks the 25% area variance allowance.
1. Check all faces. If any face area exceeds $a_{max}$ (default 25% of $\bar{A}$), perform longest-edge bisection.
2. Because the variance tolerance is high, strict geometry is preserved. **Do not apply Laplacian smoothing** to new vertices. The relaxed area targets naturally prevent the deep recursive subdivision that causes severe slivers.

### Stage 4 — Dual graph construction
1. Node weight = face area.
2. Surface edge cost = $\|c_i - c_j\| \times (1 + \mu \cdot \text{concavity}(e))$. Concave creases cost more to cross, trapping seams in folds.
3. Insert bridge edges from Stage 2.

### Stage 5 — Seeding
Place $n_k$ deterministic seeds per major component using graph-metric farthest-point sampling (multi-source Dijkstra).

### Stage 6 — Capacity-constrained growth
Single global priority queue over all seeds. The key for claiming face $f$ into region $i$:

$$key = d_i(f) \cdot \left( \frac{\max(A_i, \zeta \cdot \bar{A})}{\bar{A}} \right)^{\alpha_i}$$

*   **Lazy-Evaluation Fix:** Because $A_i$ increases during growth, keys become stale. Store `(key, face_id, region_id)`. On pop, recalculate the key using current $A_i$. If worse, push back into the queue and grab the next item.

### Stage 7 — Repair (Atomic Dump)
1.  **Satellite fixup:** Force every satellite's cells to its anchor's label.
2.  **Connectivity (The Dump):** Compute connected components of each label. Keep the largest by area. Donate the orphaned remainder cells *atomically* to the adjacent region with the smallest area. Because $\epsilon_{bal}$ is relaxed to 25%, this dump will rarely trigger runaway elongation penalties.
3.  **Compactness:** Compute the 3D covariance matrix of the region's face centroids. If the eigenvalue ratio $\rho_i > \gamma$, raise $\alpha_i$ for the next pass to prevent ribbon-like pieces.
4.  Re-seed any emptied regions.

### Stage 8 — Lloyd relaxation
1. Re-seed at region medoids, re-run Stage 6, re-run Stage 7.
2. Score the iteration based on area deviation, seam length, and elongation.
3. Stop when $\max(|A_i - \bar{A}|) / \bar{A} \le \epsilon_{bal}$ (25%), or $I_{max}$ is reached. Retain the best-scoring iteration.

### Stage 9 — Seam fairing (UV-Strict)
*Purpose:* Smooth the jagged staircase edges of regions into clean curves without destroying authored data.
1. Chain seam edges into polylines.
2. **Strict Attribute Locking:** Identify junction vertices (where 3+ labels meet). **Identify all edges that lie on a UV seam or hard normal boundary.** These are permanently pinned.
3. Shorten segments between pinned endpoints using band min-cut or flip geodesics.
4. **No UV Tearing:** The fairing algorithm is strictly forbidden from flipping or cutting across a locked UV edge. The seam will remain jagged at these exact intersections to perfectly preserve the original texture map.

### Stage 10 — Extraction
1. Gather faces per label, compact vertex buffers.
2. Preserve corner attributes (UVs/normals) verbatim.
3. Write meshes and JSON manifest.

---

## 8. Library Stack Directives
*   **Python Prototype:** `trimesh`, `libigl` (bindings for `fast_winding_number`), `scipy.sparse.csgraph`. Use `bpy.types.Mesh.foreach_get` for zero-overhead Blender data ingestion.
*   **C++ Production Core:** `geometry-central` (half-edge, robust on soup, intrinsic geodesics), `libigl`, `nanoflann` (for Stage 2 BVH/proximity).

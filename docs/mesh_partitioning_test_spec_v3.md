# Surface-area-balanced mesh partitioning - Validation Plan v3.0

## Required Test Fixtures
To execute these tests, you must author or source the following minimalist 3D assets. Keep their polygon counts deliberately low (under 5k) so you can step through the data in a debugger if needed.

*   **Fixture A (The Sphere):** A perfect, watertight UV-mapped sphere with roughly uniform triangulation.
*   **Fixture B (The Soup):** An open disc with a hole in the middle, containing non-manifold edges (a "T-junction" where 3 faces share an edge), and at least one degenerate face (zero area).
*   **Fixture C (The Shirt):** A torso mesh with three disconnected "button" meshes floating slightly in front of it. 
*   **Fixture D (The Gem):** A ring mesh with a disconnected gem mesh nested inside the setting.
*   **Fixture E (The Ribbon):** A long, flat, thin rectangle.
*   **Fixture F (The UV Box):** A cube with distinct UV islands on each face and hard normal creases on every edge.

---

## Stage 0: Ingest and Conditioning
**Objective:** Prove that the working mesh is topologically sound without destroying authored attributes (A4).

| ID | Test Target | Fixture | Validation Procedure | Pass Condition |
| :--- | :--- | :--- | :--- | :--- |
| **0.1** | Area Conservation | All | Sum face areas before and after conditioning (excluding removed degenerates). | Delta is less than 1e-6 relative. |
| **0.2** | Degenerate Purge | B (Soup) | Count faces with area below the threshold before and after. | Post-ingest count is exactly 0. |
| **0.3** | Attribute Survival | F (UV Box) | Count UV islands and hard edges before and after position welding. | Exact match. No UV seams are merged. |
| **0.4** | Non-Manifold Survival | B (Soup) | Load the mesh and build the half-edge structure. | Does not crash; T-junctions are recorded as adjacent face pairs. |
| **0.5** | Determinism | All | Load the same fixture twice. Compare vertex buffers and face arrays. | Byte-identical. |

---

## Stage 1: Component Triage
**Objective:** Prove the algorithm correctly identifies major components versus satellites, and distributes the target piece count appropriately.

| ID | Test Target | Fixture | Validation Procedure | Pass Condition |
| :--- | :--- | :--- | :--- | :--- |
| **1.1** | Classification | C (Shirt) | Verify component labels. | Torso is flagged as Major. Buttons are flagged as Satellites. |
| **1.2** | Allocation Math | A (Sphere) | Request N=5 on a single major component. | Component receives exactly 5 pieces. |
| **1.3** | Infeasibility Catch | C (Shirt) | Request N=1 (fewer than the number of major components). | Pipeline safely raises an error stating N is too low. |

---

## Stage 2: Bridge Construction
**Objective:** Prove satellites attach to the correct visual host without spanning illegal gaps.

| ID | Test Target | Fixture | Validation Procedure | Pass Condition |
| :--- | :--- | :--- | :--- | :--- |
| **2.1** | Gap Cap Limit | C (Shirt) | Move a button 10 meters away from the shirt. Run bridge construction. | Button finds no bridge and is promoted to a Major component. |
| **2.2** | Normal Agreement | C (Shirt) | Place a button on the front of the shirt. Verify anchor cell. | Anchors to the front panel, ignoring the back panel (even if physically closer). |
| **2.3** | Enclosure Override | D (Gem) | Run bridge construction on the nested gem. | Gem anchors to the ring setting (winding number > 0.5), not the outside floor. |
| **2.4** | Anchor Termination | C (Shirt) | Trace the bridge chain from the button. | Bridge chain terminates on a cell belonging to a Major component. |

---

## Stage 3: Refinement (Optimized)
**Objective:** Prove large faces are bisected to prevent area imbalances without altering the original shape.

| ID | Test Target | Fixture | Validation Procedure | Pass Condition |
| :--- | :--- | :--- | :--- | :--- |
| **3.1** | Max Area Cap | A (Sphere) | Subdivide using an artificially small max area parameter. | No face in the working mesh exceeds the parameter. |
| **3.2** | T-Junction Prevention | A (Sphere) | Count incident faces per edge after bisection. | No newly created edge has more than 2 incident faces. |
| **3.3** | Strict Geometry | A (Sphere) | Compare surface volume or total area before and after bisection. | Exactly identical. No Laplacian smoothing occurred. |
| **3.4** | Attribute Interpolation | F (UV Box) | Sample UV coordinates at newly inserted bisection vertices. | Values are mathematically continuous with the original face UVs. |

---

## Stage 4: Dual Graph Construction
**Objective:** Prove the underlying navigation network for the algorithm is connected and weighted correctly.

| ID | Test Target | Fixture | Validation Procedure | Pass Condition |
| :--- | :--- | :--- | :--- | :--- |
| **4.1** | Symmetry | All | Scan the CSR adjacency lists. For every edge U->V, check V->U. | Edge exists in both directions with exactly equal cost. |
| **4.2** | Positive Costs | All | Iterate all dual graph edge costs. | No zero, negative, NaN, or infinite costs. |
| **4.3** | Concavity Weighting | F (UV Box) | Compare edge costs across a flat face vs across a 90-degree internal corner. | The concave corner edge cost is strictly higher than the flat edge cost. |
| **4.4** | Augmented Connectivity | C (Shirt) | Check connected components of the dual graph including bridges. | Results in exactly one connected component. |

---

## Stage 5: Seeding
**Objective:** Prove the starting points for region growth are spread far apart and are 100% deterministic.

| ID | Test Target | Fixture | Validation Procedure | Pass Condition |
| :--- | :--- | :--- | :--- | :--- |
| **5.1** | Count | All | Count generated seeds. | Matches N exactly. |
| **5.2** | Distinctness | All | Check seed IDs for duplicates. | All seeds are unique cells. |
| **5.3** | Determinism | A (Sphere) | Run the seeder 10 times with the same parameters. | Yields the exact same cell indices every run. |

---

## Stage 6: Capacity-Constrained Growth
**Objective:** Prove the core priority queue correctly claims faces without stranding geometry or failing to update costs.

| ID | Test Target | Fixture | Validation Procedure | Pass Condition |
| :--- | :--- | :--- | :--- | :--- |
| **6.1** | Lazy-Evaluation Queue | A (Sphere) | Log the priority key of faces when pushed vs when popped. | No face is claimed using a stale key (area size is verified on pop). |
| **6.2** | Total Coverage | All | Count unlabelled cells at exhaustion. | Exactly 0. Every face belongs to a region. |
| **6.3** | Soft Cap Integrity | E (Ribbon) | Request N=2 on a shape that forces one region to over-consume. | Pipeline completes without raising an error; no cells are orphaned. |
| **6.4** | Area Accounting | All | Sum the tracked areas of all regions. | Matches the total mesh area (Delta < 1e-6). |

---

## Stage 7: Repair (Atomic Dump)
**Objective:** Prove the algorithm fixes disconnected puzzle pieces and correctly penalizes ribbon-like shapes.

| ID | Test Target | Fixture | Validation Procedure | Pass Condition |
| :--- | :--- | :--- | :--- | :--- |
| **7.1** | Satellite Coherence | C (Shirt) | Compare the label of a button to the label of its anchor cell on the shirt. | Labels are identical. |
| **7.2** | Connectivity | A (Sphere) | Run connected components per label over the dual graph. | Every label consists of exactly 1 connected component. |
| **7.3** | Atomic Dump Logic | A (Sphere) | Force a split region. Log the area of the neighbor that receives the orphan island. | The receiving neighbor was the one with the smallest area. |
| **7.4** | Eigenvalue Elongation | E (Ribbon) | Log the elongation metric on a very long, straight region. | The metric exceeds the threshold and the penalty exponent is raised. |

---

## Stage 8: Lloyd Relaxation
**Objective:** Prove the loop pushes regions toward the 25% area variance target and terminates gracefully.

| ID | Test Target | Fixture | Validation Procedure | Pass Condition |
| :--- | :--- | :--- | :--- | :--- |
| **8.1** | Convergence | A (Sphere) | Log the max area deviation after the loop finishes. | Maximum deviation is under 25%. |
| **8.2** | Best Retained | A (Sphere) | Compare the score of the final returned labelling against the lowest score logged during the loop. | The final state matches the best state, even if the final iteration was worse. |
| **8.3** | Termination Limits | E (Ribbon) | Run on an impossible shape that will never balance. | Halts strictly at the maximum iteration limit without infinite looping. |

---

## Stage 9: Seam Fairing (UV-Strict)
**Objective:** Prove that regions are visually smoothed but perfectly preserve UVs and normal maps.

| ID | Test Target | Fixture | Validation Procedure | Pass Condition |
| :--- | :--- | :--- | :--- | :--- |
| **9.1** | Strict Attribute Locking | F (UV Box) | Verify the vertex indices along the UV boundaries before and after fairing. | Exact match. Locked edges were not flipped, moved, or deleted. |
| **9.2** | Length Reduction | A (Sphere) | Sum the total polyline length of all seams before and after fairing. | Total length is strictly shorter. |
| **9.3** | Safe Re-triangulation | A (Sphere) | Sample normals inside a face that was cut by a new seam vertex. | Normals interpolate perfectly across the cut. |

---

## Stage 10: Extraction
**Objective:** Prove the generated pieces are ready for export and physically reassemble into the exact original model.

| ID | Test Target | Fixture | Validation Procedure | Pass Condition |
| :--- | :--- | :--- | :--- | :--- |
| **10.1** | Reassembly (The Final Boss) | All | Load all extracted pieces, merge their vertex buffers based on their saved rest poses. | Hausdorff distance to the original source mesh is below 1e-5. |
| **10.2** | Mesh Cleanliness | All | Scan extracted pieces for unreferenced vertices or degenerate faces. | Zero unreferenced vertices, zero degenerates. |
| **10.3** | Material Integrity | F (UV Box) | Verify the material index arrays on the extracted pieces. | No materials are lost, misassigned, or swapped. |

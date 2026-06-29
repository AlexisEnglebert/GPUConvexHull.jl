# GPUConvexHull — Possible Improvements

This document collects concrete improvement directions for `GPUConvexHull.jl`,
derived from (a) a review of the current implementation and (b) a study of why
[Qhull](http://www.qhull.org/) remains competitive even **CPU-vs-CPU**, and
especially why it scales far better in dimensions `d ≥ 4`.

The reference Qhull source studied here is the reentrant library
`qhull/src/libqhull_r/` (file/function/line citations below point there).

> **TL;DR.** The gap to Qhull is *not* primarily floating-point speed or
> language. Qhull is **output-sensitive and local** at every step (outside-sets,
> greedy graph descent, hash-matched neighbors, pooled memory, facet merging),
> while the current implementation is **global and cumulative** (all-points ×
> all-faces classification, `O(F²)` neighbor recomputation, no merging, no
> reclamation of dead faces). The high-dimensional memory blow-up is a direct
> consequence of the *cumulative* design, not of per-call allocation hygiene.

---

## 0. Diagnosis: why memory explodes in 5D/6D

It is tempting to blame the high-dimensional slowdown on per-iteration GPU
buffer churn (reallocating `distances`, `compact_mask`, `cand_idx`, … every
loop). That is real but it is a **constant-factor / GC-pressure** issue, *not*
the cause of the dimensional explosion. Three compounding effects are:

1. **Simplicial Upper Bound Theorem.** A simplicial `d`-polytope on `n` vertices
   has `Θ(n^⌊d/2⌋)` facets — `O(n²)` in 4D/5D, `O(n³)` in 6D. Every face here is
   a simplex by construction, so we pay the worst case of the bound.
2. **No merging ⇒ vertex inflation.** With a fixed `ε`, every near-coplanar
   point is promoted to a real vertex, and each extra vertex spawns its own cone
   of simplices. This is exactly the observed `381 vs 307` vertices in 6D
   (validation table): ~24% more vertices than Qhull, and the gap widens with
   `d`.
3. **Dead faces are never reclaimed.** Removed faces are only flagged
   `active = false`; they are never compacted out of `normals` / `offsets` /
   `neighbors`. Memory therefore tracks **cumulative** faces ever created, not
   the **live** hull, and `findall(mesh.active)` / `sum(mesh.active)` are swept
   `O(total-ever)` every iteration.

**Consequence:** buffer reuse alone would *not* fix 5D. The levers that matter,
in order, are: **(a) merge coplanar facets**, **(b) compact out inactive
faces**, and only then **(c) reuse scratch buffers**.

---

## 1. Algorithmic gaps vs Qhull (the asymptotic wins)

These are the differences that make Qhull win even on CPU.

### 1.1 Point→facet classification: locality instead of `O(n·F)`

- **Current:** `assignFaceToPointKernel` tests each point against *every* face
  (`for face = 1:size(G_normal, 2)`), i.e. `O(n·F)` per (re)assignment.
- **Qhull:** `qh_findbest` (`geom_r.c:156`) does a **greedy steepest-ascent walk
  over the facet adjacency graph** — only `O(dim)` neighbors per hop, with a
  `visitid` stamp to avoid re-tests and a `MINoutside` early-out. Cost ≈
  `O(dim · path-length)`, not `O(F)`. Outside-sets keep each point in exactly
  one facet's bucket, with the furthest cached (`furthestdist`,
  `qh_setdellast` is `O(1)`); only points of *deleted* facets are ever
  re-partitioned, and only against the **local** new cone
  (`qh_partitionvisible`, `libqhull_r.c:1454`).
- **Improvement:** even a partial port of this idea (descend the `neighbors`
  graph from a point's current face rather than scanning all faces) removes the
  dominant `O(n·F)` term. Note this is serial/divergent and GPU-hostile in its
  pure form → a **hybrid** (GPU for the bulk distance passes, local descent for
  reassignment) is the realistic target. **Open research direction.**

### 1.2 Neighbor maintenance: hashing instead of `O(F²)`

- **Current:** after creating new faces, neighbors are recomputed by comparing
  every new face against every other new face (`intersect` / `findfirst` over
  vertex vectors) → `O(F²)` (acknowledged in the thesis).
- **Qhull:** `qh_matchnewfacets` (`poly_r.c:937`) hashes each new facet's ridges
  (vertex-set-minus-one-vertex) into a table and cross-links matching facets in
  `O(1)` each via `qh_matchneighbor` (`poly_r.c:818`) → `O(F·dim)` expected.
- **Improvement:** replace the `O(F²)` loop with **ridge hashing**. This is a
  pure constant/asymptotic win and is feasible on CPU immediately; a GPU hash
  is a follow-up. This is *not* just "fewer set-differences" — it is an
  algorithmic change.

### 1.3 Horizon / visibility: local flood-fill

- **Qhull:** `qh_findhorizon` (`libqhull_r.c:803`) expands only across neighbor
  links from the seed facet, touching the visible region + its one-ring horizon
  — never all facets.
- **Current:** the BFS (`get_visible_faces`) is the right idea but runs on the
  **CPU** per candidate per iteration, on `Vector{Vector}` topology. Moving it
  (and the horizon/ridge detection) to the GPU is listed below.

### 1.4 Custom pooled allocator

- **Qhull:** `qh_memalloc` (`mem_r.c:92`) pops size-class **free lists** in
  `O(1)`; large buffers are `malloc`'d rarely. Facet/vertex/ridge/set churn
  during cone build-and-delete is nearly free.
- **Current:** relies on the KA backend memory pool for scratch (fine) but the
  growing `mesh.*` arrays are never recycled (see §0.3).

---

## 2. Robustness & facet-count gaps (what closes 381→307 and stabilises `d ≥ 4`)

### 2.1 Facet merging / thick-facet model — the biggest structural gap

- Qhull merges near-coplanar / slightly-concave facets into a single
  **non-simplicial** facet with an inner/outer plane (`qh_premerge`,
  `merge_r.c:75`; `qh_all_merges`, `merge_r.c:223`; `qh_mergefacet`,
  `merge_r.c:3361`). This is *why* it reports fewer facets and vertices, and why
  it does not triangulate flat regions.
- **Improvement:** add a coplanar/concave merge pass with a thick-facet
  representation. This is the single biggest lever on both **output size** and
  **high-d memory** (it shrinks the face count that drives §0).

### 2.2 Adaptive, dimension- and magnitude-relative tolerance

- **Current:** fixed absolute `ε = 1e-9`.
- **Qhull:** `qh_distround` (`geom2_r.c:408`):
  `DISTround ≈ ε_mach·(d^1.5·maxabs·1.01 + maxabs)`, from a `qh_maxmin` scan of
  the data, plus a per-pivot `NEARzero[k] = 80·MAXsumcoord·ε_mach`
  (`geom2_r.c:1160`).
- **Worked example (this repo's uniform benchmark, integer points in
  `[-10⁶, 10⁶]`):** at `maxabs = 10⁶, d = 6`, Qhull's `DISTround ≈ 3.5e-9`,
  while the fixed `ε = 1e-9` is ~**3.5× tighter** — consistent with promoting
  extra near-coplanar points to vertices (381 vs 307). The fixed value happens
  to land in the right *order of magnitude* for `10⁶`-scale data **by luck**;
  rescale the inputs by `10⁶` and it becomes absurdly tight, by `10⁻⁶` and it
  becomes meaningless. Tolerance must track data magnitude **and** dimension.
- **Improvement:** compute `ε` adaptively from a one-pass coordinate min/max and
  the dimension. Low effort, high robustness payoff; partially closes the vertex
  gap independently of merging.

### 2.3 Redundant-vertex elimination

- Even with merging, Qhull drops vertices that become interior to a merged thick
  facet (`qh_reducevertices` / `qh_test_redundant_neighbors`,
  `merge_r.c:965`). Needed to actually reach Qhull's 307-vertex count.

### 2.4 Robust facet normals via pivoted Gaussian elimination

- **Current:** SVD/LAPACK per face.
- **Qhull:** closed-form determinants for `d ≤ 4` (`qh_sethyperplane_det`),
  Gaussian elimination with **partial pivoting** for higher `d`
  (`qh_gausselim`, `geom_r.c:620`; `qh_backnormal`, `geom_r.c:560`), with a
  `nearzero` flag and an **interior-point orientation cross-check**
  (`qh_orientoutside`). Benefits beyond speed: orientation sign falls out of the
  pivot swaps, relative pivot thresholds give backward stability, and there is
  no per-facet allocation (reuses global buffers).

### 2.5 Well-conditioned initial simplex

- **Current:** min/max along each axis + orthogonal-projection residual to fill
  missing points; the concern is "did we get `d+1` **distinct** points."
- **Qhull:** `qh_maxsimplex` (`geom2_r.c`) greedily **maximises the
  determinant** one vertex at a time (threshold `qh_RATIOmaxsimplex = 1e-3`).
- **Improvement:** select the simplex by conditioning, not mere distinctness. A
  near-coplanar start cascades into flipped facets and forced merges in high `d`.

### 2.6 Topology-repair for high-d roundoff failure modes

- Qhull explicitly repairs **flipped facets** (`qh_flippedmerges`), **pinched
  horizons / duplicate ridges** (`qh_mark_dupridges`, `qh_forcedmerges`), and
  degenerate/redundant facets, with bounded wide-merge guards
  (`qh_WIDEdupridge`, `qh_WIDEduplicate`). These occur from roundoff even on
  "nice" inputs once `d ≥ 4`. The current code has no repair mechanism — a
  robustness blind spot to be aware of.

### 2.7 Joggle-and-restart fallback (alternative to merging)

- If a purely **simplicial** output is desired, Qhull's other robustness
  strategy is `qh_joggleinput` (perturb to general position, magnitude
  `DISTround·30000`) with `qh_joggle_restart` on any precision failure. This is
  GPU-friendly (one perturbation pass + retry) and avoids the entire
  thick-facet machinery, at the cost of a controlled perturbation of the result.

---

## 3. Engineering / code-quality improvements

Independent of the algorithmic work above:

- **Compact out inactive faces** from `mesh.normals/offsets/neighbors` so memory
  tracks the live hull (see §0.3). Highest-value memory fix.
- **Reuse per-iteration GPU scratch buffers** (`distances`, `compact_mask`,
  `cand_idx`, `face_*_gpu`, …) instead of reallocating each loop. Helps the
  2D/3D constant factor and GC pressure.
- **Factor the duplicated compact-permute block** (it appears ~twice and has
  already drifted, e.g. `zeros` vs `allocate`) into a single helper.
- **Generalise or document `prepare_output`**: the final hull-point ordering
  uses `atan` on dims 1–2, so the *ordering* is only meaningful in 2D
  (`hull_indices` is general). Either generalise or clearly scope it.
- **Implement the shared-memory bank-conflict padding** that the scan kernels
  are described as using (the `i_pad = i + ⌊i/32⌋` remapping), or correct the
  claim — the committed kernels currently use raw indexing with no
  `LOG_NUM_BANKS` offset.
- **Dimension-specialise the distance kernel** (Qhull's `qh_distplane`,
  `geom_r.c:50`, unrolls the dot product for dims 2–8). Normals are already
  cached on the face; specialise the hot inner product.
- **Cleanup:** remove dead/commented code and informal comments, fix typos
  (`kernell`, `inital_value`, `forwad_scan`, `Downweep`), and seed the
  randomised tests.

---

## 4. Testing / validation improvements

- **Add a Qhull oracle** to the end-to-end tests (e.g. via `Qhull_jll` /
  `MiniQhull` / `Polyhedra`) instead of hand-computed expected hulls. Currently
  only the *primitives* (scan, min/max) are checked against an independent
  reference; the hull tests are small hand-built fixtures.
- **Extend coverage to 5D/6D** (the suite stops at 4D; the thesis claims 2D–6D).
- **Exercise GPU backends in CI** or document that CI is CPU-only. Today every
  test runs on `CPU()`; CUDA/oneAPI portability is asserted but not
  automatically verified, and no `oneAPI` reference appears in the repo.
- **Check topology, not just vertex sets** (facets / orientation), which matters
  most for the higher-dimensional correctness claims.
- **Resolve or track the parked cases** (the commented-out "all points on edges"
  test and the flaky 3D-cube case).

---

## 5. Suggested ordering

A pragmatic sequence that front-loads the wins relevant to the 5D blow-up:

1. **Compact out dead faces** (§3, §0.3) — biggest memory win, low risk.
2. **Adaptive `ε`** (§2.2) — low effort, closes part of the vertex gap, improves
   robustness at all scales.
3. **Ridge-hashed neighbor matching** (§1.2) — removes the `O(F²)` term on CPU.
4. **Reuse scratch buffers** (§3) — constant-factor / GC win.
5. **Facet merging + redundant-vertex elimination** (§2.1, §2.3) — the large
   structural change that closes 381→307 and tames high-d face counts.
6. **Local point→facet descent** (§1.1) and **GPU horizon/visibility** — the
   harder, more open GPU-research items.
7. **Robust normals (pivoted GE) + well-conditioned simplex + topology repair**
   (§2.4–§2.6) — needed to stay correct as `d` grows.

Items 1–4 are mostly engineering and immediately actionable; items 5–7 are the
research contributions that would make the implementation genuinely competitive
with Qhull in `d ≥ 4`.

---

### Reference map (Qhull `libqhull_r`)

| Concept | Function | File:line |
|---|---|---|
| Greedy best-facet descent | `qh_findbest` | `geom_r.c:156` |
| Local partition of orphans | `qh_partitionvisible` | `libqhull_r.c:1454` |
| Hash-matched neighbors | `qh_matchnewfacets` / `qh_matchneighbor` | `poly_r.c:937` / `818` |
| Local horizon | `qh_findhorizon` | `libqhull_r.c:803` |
| Pooled allocator | `qh_memalloc` / `qh_memfree` | `mem_r.c:92` / `227` |
| Facet merging | `qh_premerge` / `qh_all_merges` / `qh_mergefacet` | `merge_r.c:75` / `223` / `3361` |
| Redundant-vertex removal | `qh_reducevertices` | `merge_r.c:965` |
| Adaptive distance tolerance | `qh_distround` / `qh_detroundoff` | `geom2_r.c:408` / `225` |
| Pivot tolerance | `NEARzero[k]` | `geom2_r.c:1160` |
| Normals (det / Gauss) | `qh_sethyperplane_det` / `qh_gausselim` | `geom_r.c:1152` / `620` |
| Initial simplex by det | `qh_maxsimplex` | `geom2_r.c` |
| Distance kernel (unrolled) | `qh_distplane` | `geom_r.c:50` |
| Joggle | `qh_joggleinput` / `qh_joggle_restart` | `geom2_r.c:995` / `libqhull_r.c:898` |

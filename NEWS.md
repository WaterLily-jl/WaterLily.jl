# WaterLily.jl release notes

## v1.10.0

WaterLily v1.10 is out. It follows v1.9 by only two weeks, so this note covers everything since **v1.8** (June 2026): 42 pull requests from 6 contributors.

### Performance since v1.8

Time per `sim_step!` (mean, ms) and speedup over v1.8.0:

| Case | Backend | v1.8.0 | v1.9 | v1.10 |
|---|---|---|---|---|
| Taylor-Green vortex, 64³ | CPU, 4 threads | 33.5 | 17.7 (1.89×) | **14.4 (2.33×)** |
| Moving cylinder, 0.44 M cells | CPU, 4 threads | 56.5 | 41.8 (1.35×) | **38.9 (1.45×)** |
| Jellyfish, Biot-Savart BCs, 0.13 M cells | CPU, 4 threads | 73.9 | 50.9 (1.45×) | 51.9 (1.42×) |
| Taylor-Green vortex, 128³ | GPU | 43.7 | 36.9 (1.19×) | **30.8 (1.42×)** |
| Moving cylinder, 3.5 M cells | GPU | 82.3 | 75.8 (1.09×) | **69.3 (1.19×)** |
| Jellyfish, Biot-Savart BCs, 1.0 M cells | GPU | 79.2 | 72.5 (1.09×) | **62.8 (1.26×)** |

How these were measured:
- **Hardware:** a Windows laptop with an NVIDIA RTX A2000 8GB Laptop GPU; Julia 1.11.9, Float32.
- **Method:** [WaterLily-Benchmarks](https://github.com/WaterLily-jl/WaterLily-Benchmarks), timing 5 runs of 25 steps from a developed flow (minimum of the run means).
- **Jellyfish:** each WaterLily version runs with the BiotSavartBCs.jl of the same time, so these rows include BiotSavartBCs changes too.
- **Moving bodies in v1.10:** the cylinder and jellyfish use the new `WaterLilyNarrowBand.jl` (see below).
- **v1.9 on GPU:** measured with the v1.9.0 code plus the `mom_project!` GPU fix released in v1.9.1, since v1.9.0 itself doesn't run on GPU.

### Highlights

- **A faster core solver.** Each step now makes fewer passes over memory, because `conv_diff!`, `BDIM!` and `residual!` are fused and vectorized ([#354](https://github.com/WaterLily-jl/WaterLily.jl/pull/354)). The QUICK limiter no longer branches ([#326](https://github.com/WaterLily-jl/WaterLily.jl/pull/326)), which matters on GPUs. Together with the solver changes below, this gives the speedups above.
- **Grid-independent pressure convergence** ([#307](https://github.com/WaterLily-jl/WaterLily.jl/pull/307)). The Poisson solver now stops when both the mean residual (L₁/N) and the maximum residual (L∞) pass `tol`, so convergence no longer depends on grid size. The default `tol` changed from `1e-4` to `2e-3` to match. ⚠️ If you pass `tol` yourself, check the value.
- **Bodies can limit their own work** ([#361](https://github.com/WaterLily-jl/WaterLily.jl/pull/361), [#363](https://github.com/WaterLily-jl/WaterLily.jl/pull/363)). A body can overload `WaterLily.bbox(body, R)` to return the region where it needs measuring. `measure!` and the force and moment integrals then only loop over that region. The default is the whole domain, so nothing changes unless a body opts in.
- **[WaterLilyNarrowBand.jl](https://github.com/WaterLily-jl/WaterLilyNarrowBand.jl)**, a new companion package built on this hook. `NarrowBand(body, dims)` wraps any `AbstractBody` so it is only evaluated in a thin band around its surface, giving the same BDIM fields and forces at a fraction of the measurement cost.
- **Correct pressure for deforming bodies** ([#328](https://github.com/WaterLily-jl/WaterLily.jl/pull/328), [@marinlauber](https://github.com/marinlauber)). The pressure source now excludes the BDIM stretching term from the divergence.
- **More GPUs:** Apple Metal now works ([#327](https://github.com/WaterLily-jl/WaterLily.jl/pull/327), [#329](https://github.com/WaterLily-jl/WaterLily.jl/pull/329), [@timesselens](https://github.com/timesselens)), alongside fixes for AMDGPU ([#348](https://github.com/WaterLily-jl/WaterLily.jl/pull/348)) and CUDA ([#338](https://github.com/WaterLily-jl/WaterLily.jl/pull/338), [@KristianHolme](https://github.com/KristianHolme)). GPU tests now run on every PR, on the JuliaGPU Buildkite cluster ([#342](https://github.com/WaterLily-jl/WaterLily.jl/pull/342)).

### Details

**Solver**
- **Boundary conditions** ([#339](https://github.com/WaterLily-jl/WaterLily.jl/pull/339)): now enforced in `mom_project!`.
- **CFL time step** ([#352](https://github.com/WaterLily-jl/WaterLily.jl/pull/352)): taken from the interior cells only.
- **Red-black Gauss-Seidel** ([#353](https://github.com/WaterLily-jl/WaterLily.jl/pull/353)): smooths the last interior plane for odd grid sizes.
- **`MultiLevelPoisson`** ([#308](https://github.com/WaterLily-jl/WaterLily.jl/pull/308)): no longer allocates on every V-cycle.
- **Kernel launches** ([#330](https://github.com/WaterLily-jl/WaterLily.jl/pull/330), [#345](https://github.com/WaterLily-jl/WaterLily.jl/pull/345), [#346](https://github.com/WaterLily-jl/WaterLily.jl/pull/346)): `@loop` launches are type-stable, boundary loops launch over the faces only, and the SIMD loop uses `@inbounds`.

**Bodies and metrics**
- **`RigidMap(center)`** ([#357](https://github.com/WaterLily-jl/WaterLily.jl/pull/357)): the rotation now defaults to zero.
- **Force and moment integrals** ([#361](https://github.com/WaterLily-jl/WaterLily.jl/pull/361)): these now sum in the element type of the fields.
- **Helicity density** ([#292](https://github.com/WaterLily-jl/WaterLily.jl/pull/292), [@TzuYaoHuang](https://github.com/TzuYaoHuang)): new `WaterLily.helicity(I, u, ω)`.
- **Combined bodies** ([#368](https://github.com/WaterLily-jl/WaterLily.jl/pull/368)): `body.map` returns the shared map of a `SetBody`.

**GPU**
- **Boundary and initial conditions** ([#330](https://github.com/WaterLily-jl/WaterLily.jl/pull/330)): `uBC` and `u0` tuples work on GPU.
- **Metal support** ([#329](https://github.com/WaterLily-jl/WaterLily.jl/pull/329)): `L₂` is backend-generic, and `ke` and `ω_θ` keep the field's element type.

**Flow and I/O**
- **`Flow` constructor** ([#340](https://github.com/WaterLily-jl/WaterLily.jl/pull/340)): new field-wise constructor, so `setproperties(flow; uBC)` swaps boundary conditions without reallocating the fields.
- **`load!`** ([#314](https://github.com/WaterLily-jl/WaterLily.jl/pull/314)): copies host arrays straight into place, with no transient device copy.
- **`interp`** ([#371](https://github.com/WaterLily-jl/WaterLily.jl/pull/371)): vector interpolation is 3-4x faster on GPU, speeding up Pathlines.jl's `Particles`.

**Visualization**
- **`viz!`** ([#334](https://github.com/WaterLily-jl/WaterLily.jl/pull/334)): new `isomesh` keyword for solid 3D isosurfaces.
- **`sim_gif!`** ([#323](https://github.com/WaterLily-jl/WaterLily.jl/pull/323), [@TzuYaoHuang](https://github.com/TzuYaoHuang)): takes the same field and plotting options as `viz!`.
- **Vector export** ([#319](https://github.com/WaterLily-jl/WaterLily.jl/pull/319)): volume plots are rasterized when exporting to SVG and PDF.
- **Pathlines** ([#313](https://github.com/WaterLily-jl/WaterLily.jl/pull/313), [#336](https://github.com/WaterLily-jl/WaterLily.jl/pull/336)): the Pathlines-based interactive `viz!`/`viz_step!` backend now lives in Pathlines.jl.

**Development**
- **Benchmarking PRs** ([#350](https://github.com/WaterLily-jl/WaterLily.jl/pull/350)): any PR can be benchmarked on demand with a `/benchmark` comment.

### API changes across the ecosystem

For developers of packages built on WaterLily. The WaterLily version that introduced each change is in brackets, for setting `[compat]` bounds.

**WaterLily.jl**
- **`mom_project!(a, b, w::Int, t)`** [1.9.1] ([#339](https://github.com/WaterLily-jl/WaterLily.jl/pull/339)): now applies the domain BCs (and `exitBC!` in the predictor) before the pressure solve, and divides the time step by `w` (1 in the predictor, 2 in the corrector). Overrides of `mom_predict!` and `mom_correct!` should no longer call `BC!` or `exitBC!`, and custom `mom_project!` methods need the new signature.
- **Pressure source** [1.9.1] ([#328](https://github.com/WaterLily-jl/WaterLily.jl/pull/328), [#338](https://github.com/WaterLily-jl/WaterLily.jl/pull/338)): `div(I, u) - δv(D, I)*div(I, V)`, where `D = diagonal(pois)` is a plain array, safe to pass into GPU kernels. Custom Poisson solvers that set their own source should match.
- **Poisson convergence** [1.9.0] ([#307](https://github.com/WaterLily-jl/WaterLily.jl/pull/307)): `solver!` defaults to `tol=2e-3` and stops when `L₁(p) < l1n_tol(p, tol)` and `L∞(p) < tol`. Custom solvers can reuse `L₁`, `L∞` and `l1n_tol` to match.
- **`bbox(body, R)`** [1.9.2] ([#361](https://github.com/WaterLily-jl/WaterLily.jl/pull/361)): overload for a custom `AbstractBody` to restrict `measure!` and the force integrals to the region it returns; the default returns `R`. From 1.10, `measure!` also clears the BDIM fields only over that region ([#363](https://github.com/WaterLily-jl/WaterLily.jl/pull/363)).
- **Force and moment integrals** [1.9.2] ([#361](https://github.com/WaterLily-jl/WaterLily.jl/pull/361)): each now also has `(flow, body)` and raw-array methods, e.g. `pressure_force(p, df, body, t)`, and there is a new `total_force_and_moment(x₀, sim)`. The sums run in the fields' element type; the Float64 accumulation of 1.9.0–1.9.1 is gone.
- **`@loop` over faces** [1.9.2] ([#346](https://github.com/WaterLily-jl/WaterLily.jl/pull/346)): `face(R, j)` and `face(dims, i, j, low)` return a face of the domain that `@loop` launches over directly.
- **`BDIM!(a, s=1)`** [1.9.2] ([#354](https://github.com/WaterLily-jl/WaterLily.jl/pull/354)): now divides the result by `s`, fusing in the corrector's averaging. `scale_u!` remains as a helper.
- **`Flow`** [1.9.1] ([#340](https://github.com/WaterLily-jl/WaterLily.jl/pull/340)): a field-wise constructor, so `setproperties` works on a `Flow`.
- **Vorticity** [1.9.2] ([#344](https://github.com/WaterLily-jl/WaterLily.jl/pull/344), [#347](https://github.com/WaterLily-jl/WaterLily.jl/pull/347)): `ω(I, u)` now also works in 2D (as a scalar) and is inlined. `helicity(I, u, ω)` is new [1.9.0] ([#292](https://github.com/WaterLily-jl/WaterLily.jl/pull/292)).
- **`SetBody` map** [1.10.0] ([#368](https://github.com/WaterLily-jl/WaterLily.jl/pull/368)): `body.map` returns the map shared by both parts.
- **EllipsisNotation** [1.10.0] ([#371](https://github.com/WaterLily-jl/WaterLily.jl/pull/371)): no longer a dependency.
- **Visualization** [1.9.0] ([#313](https://github.com/WaterLily-jl/WaterLily.jl/pull/313), [#336](https://github.com/WaterLily-jl/WaterLily.jl/pull/336)): `viz_step!` is exported, and `WaterLily._pathlines_viz_hook` lets a package take over `viz!` rendering, which is how Pathlines.jl plugs in.

**[BiotSavartBCs.jl](https://github.com/WaterLily-jl/BiotSavartBCs.jl)**: first registered releases, v1.0.0 and v1.0.1; requires WaterLily ≥ 1.9.2.
- **Exports:** the public API is `BiotSimulation(...; fmm=true, nonbiotfaces=(), symmetry=(), mem=Array)` and `BiotSavartPoisson`. The lower-level helpers (`MLArray`, `collect_targets`, `flatten_targets`, `fill_ω!`, `biotBC!`, `pflowBC!`) are no longer exported.
- **`symmetry` keyword** ([#20](https://github.com/WaterLily-jl/BiotSavartBCs.jl/pull/20)): marks faces as symmetry planes and includes the image vorticity across them. Needs `fmm=true` and no periodic direction.
- **Periodic in one direction** ([#19](https://github.com/WaterLily-jl/BiotSavartBCs.jl/pull/19)): set through WaterLily's `perdir`.
- **WaterLily alignment:** follows WaterLily's `mom_project!` signature and Poisson criterion, and takes the vorticity from `WaterLily.ω` ([#23](https://github.com/WaterLily-jl/BiotSavartBCs.jl/pull/23)). The kernels are also faster ([#14](https://github.com/WaterLily-jl/BiotSavartBCs.jl/pull/14), [@KristianHolme](https://github.com/KristianHolme)).

**[WaterLilyNarrowBand.jl](https://github.com/WaterLily-jl/WaterLilyNarrowBand.jl)**: new, v0.1.0; works with WaterLily ≥ 1.7, and uses `bbox` when it is available (≥ 1.9.2).
- **`NarrowBand(body, dims; k=4, interior, mem=Array)`:** wraps any `AbstractBody`; `setmap` and `update!` pass through the wrapper.
- **Hooks for body authors:** for bodies without a reliable sign or interior, such as mesh bodies, extend `WaterLilyNarrowBand.outside!(reached, body, t)` and `hasinterior(body)`.

**[LilyPad.jl](https://github.com/WaterLily-jl/LilyPad.jl)**: requires WaterLily ≥ 1.9.1.
- **`mom_project!` changes:** its `mom_predict!` and `mom_correct!` no longer apply BCs ([#3](https://github.com/WaterLily-jl/LilyPad.jl/pull/3)), and it uses the integer `w` ([#2](https://github.com/WaterLily-jl/LilyPad.jl/pull/2)).
- **`LilyFlow`:** carries `Flow`'s λ type parameter ([#1](https://github.com/WaterLily-jl/LilyPad.jl/pull/1)).

**[Pathlines.jl](https://github.com/WaterLily-jl/Pathlines.jl)**: requires WaterLily ≥ 1.9.
- **Rendering API:** now `PathlineCanvas`, `fade!` and `draw!`. `ParticleViz`, `PathViz`, `step!` and `notify!` are removed; `Particles` and `update!` are unchanged.
- **`viz!` backend** ([#3](https://github.com/WaterLily-jl/Pathlines.jl/pull/3)): hosts the pathline backend that used to be a WaterLily extension, so loading Pathlines.jl makes `viz!` render pathlines.

**[WaterLilyMeshBodies.jl](https://github.com/WaterLily-jl/WaterLilyMeshBodies.jl)**: no API changes. Fixes the mapping Jacobian in `measure` ([#12](https://github.com/WaterLily-jl/WaterLilyMeshBodies.jl/pull/12), [@marinlauber](https://github.com/marinlauber)).

**[ParametricBodies.jl](https://github.com/WaterLily-jl/ParametricBodies.jl)**: no changes since v1.8.

### Thanks

To everyone who contributed since v1.8: [@b-fg](https://github.com/b-fg), [@TzuYaoHuang](https://github.com/TzuYaoHuang), [@KristianHolme](https://github.com/KristianHolme), [@marinlauber](https://github.com/marinlauber), [@timesselens](https://github.com/timesselens) and [@weymouth](https://github.com/weymouth).

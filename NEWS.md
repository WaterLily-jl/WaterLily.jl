# WaterLily.jl release notes

## v1.10.0

WaterLily v1.10 is out. It follows v1.9 by only two weeks, so this note covers everything since **v1.8** (June 2026): 40 pull requests from 6 contributors.

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

**GPU**
- **Boundary and initial conditions** ([#330](https://github.com/WaterLily-jl/WaterLily.jl/pull/330)): `uBC` and `u0` tuples work on GPU.
- **Metal support** ([#329](https://github.com/WaterLily-jl/WaterLily.jl/pull/329)): `L₂` is backend-generic, and `ke` and `ω_θ` keep the field's element type.

**Flow and I/O**
- **`Flow` constructor** ([#340](https://github.com/WaterLily-jl/WaterLily.jl/pull/340)): new field-wise constructor, so `setproperties(flow; uBC)` swaps boundary conditions without reallocating the fields.
- **`load!`** ([#314](https://github.com/WaterLily-jl/WaterLily.jl/pull/314)): copies host arrays straight into place, with no transient device copy.

**Visualization**
- **`viz!`** ([#334](https://github.com/WaterLily-jl/WaterLily.jl/pull/334)): new `isomesh` keyword for solid 3D isosurfaces.
- **`sim_gif!`** ([#323](https://github.com/WaterLily-jl/WaterLily.jl/pull/323), [@TzuYaoHuang](https://github.com/TzuYaoHuang)): takes the same field and plotting options as `viz!`.
- **Vector export** ([#319](https://github.com/WaterLily-jl/WaterLily.jl/pull/319)): volume plots are rasterized when exporting to SVG and PDF.
- **Pathlines** ([#313](https://github.com/WaterLily-jl/WaterLily.jl/pull/313), [#336](https://github.com/WaterLily-jl/WaterLily.jl/pull/336)): the Pathlines-based interactive `viz!`/`viz_step!` backend now lives in Pathlines.jl.

**Development**
- **Benchmarking PRs** ([#350](https://github.com/WaterLily-jl/WaterLily.jl/pull/350)): any PR can be benchmarked on demand with a `/benchmark` comment.

### Thanks

To everyone who contributed since v1.8: [@b-fg](https://github.com/b-fg), [@TzuYaoHuang](https://github.com/TzuYaoHuang), [@KristianHolme](https://github.com/KristianHolme), [@marinlauber](https://github.com/marinlauber), [@timesselens](https://github.com/timesselens) and [@weymouth](https://github.com/weymouth).

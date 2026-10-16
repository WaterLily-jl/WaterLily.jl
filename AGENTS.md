# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Copilot, Cursor, ...) working on
WaterLily.jl. Claude Code reads this file through the `@AGENTS.md` import in `CLAUDE.md`.
Keep this file the single source; put tool-specific notes in tool-specific files.

Only add instructions and commands that have been tested in this repository. When this
file and the code disagree, the code is right: fix this file in the same PR. For scripts
that *use* WaterLily (examples, simulations), read the
[WaterLily-Examples `AGENTS.md`](https://github.com/WaterLily-jl/WaterLily-Examples/blob/main/AGENTS.md)
instead.

WaterLily.jl solves the incompressible Navier-Stokes equations on a uniform Cartesian
staggered grid, with immersed bodies through the Boundary Data Immersion Method (BDIM).
The same source runs on serial CPU, multithreaded CPU and GPUs, and the time step is
differentiable with ForwardDiff. In order of priority, changes must keep it **simple**
(about 1,800 lines in `src/`), **generic** (2D and 3D, every backend), **fast**
(allocation-free kernels) and **differentiable**.

Before editing, check `git status` and keep unrelated work. Do not commit or push unless
asked.

## Repository layout

| Path | Contents |
|------|----------|
| `src/WaterLily.jl` | module, exports, `Simulation`, `sim_step!`, `measure!(sim)`, stubs for extension functions |
| `src/core.jl` | `@loop`/`@inside`, `loc`, `δ`/`CI`/`CIj`, `inside`/`inside_u`/`slice`/`face`, `BC!`/`perBC!`/`exitBC!`, backend preference, GPU-safe `gradient`/`jacobian`/`derivative` |
| `src/Flow.jl` | `Flow`, `mom_step!` and its phases, `conv_diff!`, `BDIM!`, `CFL`, convection schemes, `accelerate!`, `udf!` |
| `src/Poisson.jl`, `src/MultiLevelPoisson.jl` | pressure solver: smoothers, `residual!`, `solver!`, geometric multigrid with semi-coarsening |
| `src/Body.jl`, `src/AutoBody.jl`, `src/RigidMap.jl` | `AbstractBody`, `measure!`, `bbox`, `SetBody`, `NoBody`, `AutoBody`, `RigidMap`, `setmap` |
| `src/Metrics.jl`, `src/util.jl` | forces, moments, vorticity, `MeanFlow`; `interp`, `sgs!`, `spread!` |
| `ext/` | package extensions (CUDA, AMDGPU, Plots, Makie, Meshing, WriteVTK, ReadVTK, JLD2) |
| `test/` | `runtests.jl` (ParallelTestRunner), one `test_*.jl` per area, shared setups in `helper.jl` |

Optional dependencies go in an extension: declare an empty function in
`src/WaterLily.jl`, add the package to `[weakdeps]` and `[extensions]` in `Project.toml`,
implement it in `ext/`. Never import an optional package from `src/`.

## Grid and indexing

Read this before touching a kernel; most bugs are a ghost-cell, staggering or
component-axis mistake.

- Every array has one ghost cell per side: `N` interior cells are stored as `N+2`, the
  interior is `2:N+1`. Loop over `inside(a)`, never the raw array.
- Staggered grid: scalars (`p`, `σ`) at cell centers, velocity component `i` on the
  lower face normal to `i`. Vector fields are component-last, `u[I,i]`.
- `loc(0,I)` is the center of cell `I` (`I .- 1.5`), `loc(i,I)` its lower face `i`. The
  physical domain spans `[0,N]` in cell units; signed-distance functions use this frame.
- `inside_u(dims,j)` is the interior of the faces normal to `j`; `inside_u(dims)` covers
  every component at once. `slice(dims,i,j)` is the plane at index `i` of axis `j`;
  `face(dims,i,j)` is the same plane as a range over the other axes, so `@loop` lays its
  workgroups along it. Use `face` for boundary loops.
- `δ(i,I)` is the unit index along `i`, `CIj(j,I,k)` replaces component `j` of `I`.

All grid computations go through `@loop expr over I ∈ R` (or `@inside a[I] = ...`). It
becomes a `@simd` loop with the `"SIMD"` backend preference, or a KernelAbstractions kernel
with the default `"KernelAbstractions"` preference, launched on the backend of the
**first array in the expression**. `R` can be any array of indices on that device. Kernel
bodies must be allocation-free (use `SVector`/`SMatrix`), type-stable and work for any
dimension (iterate with `for i in 1:n` or `ntuple`, never hard-code an axis). Fuse
statements that share data into one `@loop` with `;`. Prefer one kernel for every
backend; add a backend-specific method only where a library does better (e.g. the CUDA
`perdot` in `ext/WaterLilyCUDAExt.jl`).

## Types, GPUs and differentiation

- `mem` (`Array`, `CuArray`, `ROCArray`, `MtlArray`) selects the device and `T` (default
  `Float32`) the element type. Metal has no `Float64` at all, so nothing in a kernel may
  be `Float64` in a `Float32` run: convert captured scalars outside the kernel
  (`s = T(s)`), write `zero(T)`, `one(T)` or integers (`/2`, not `*0.5`), and pass the
  type to `loc`, as in `loc(i,I,eltype(a))`.
- No scalar indexing of device arrays from host code; reduce with `sum`, `maximum` or
  `mapreduce`, and copy to the host (`Array(a)`) before printing, plotting or writing.
- Derivatives inside the solver use WaterLily's `gradient`, `jacobian` and `derivative`
  (`src/core.jl`) with `SVector` inputs, not raw ForwardDiff: they are safe in GPU kernels
  and when nested.

## The time step

`mom_step!` (`src/Flow.jl`) saves `u⁰ = u`, zeroes `u`, then runs four phases. Read the
functions; these are the rules they rely on:

| Phase | Does | Holds afterwards |
|-------|------|------------------|
| `mom_predict!(a,t₀,t₁)` | `conv_diff!` of `u⁰` with `a.λ`, `udf!`, `accelerate!`, `BDIM!(a)` | interior of `u` updated, ghosts stale |
| `mom_project!(a,b,1,t₁)` | `BC!`, `exitBC!`, pressure solve with `Δt`, correct `u`, `BC!` | `u` divergence-free, BCs at `t₁` |
| `mom_correct!(a,t₁)` | `conv_diff!` of the projected `u`, `udf!`, `accelerate!`, `BDIM!(a,2)` | interior averaged with the predictor, ghosts stale |
| `mom_project!(a,b,2,t₁)` | as above with `Δt/2`, no `exitBC!` | `u` divergence-free, BCs at `t₁` |

- Domain BCs are applied only inside `mom_project!`. Code that runs between `BDIM!` and the
  projection (a `udf`, an override) must not read the ghosts of `u`.
- A `udf` runs before `BDIM!`. It receives the advecting velocity, `force!(flow,u,t)`
  (`u⁰` in the predictor, the projected `u` in the corrector), or is called as
  `force!(flow,t)`. It cannot set boundary values: custom wall conditions are applied
  after each `mom_project!` (see `TwoD_LidCavity.jl` in WaterLily-Examples).
- The convective fluxes `ϕu`, `ϕuP`, `ϕuL`, `ϕuR` are branchless and read the cells on both
  sides of a face; a loop range must leave a valid cell on each side.
- `flow.σ` and `flow.f` are scratch, overwritten by most solver calls. `flow.u⁰` is free
  between steps (`total_force_and_moment` uses it).
- The convection scheme `λ` is chosen when the `Simulation` is built and stored in
  `flow.λ`. `CFL` zeroes `σ` before taking the interior maximum.

## Pressure solver

- The finest-level Poisson coefficients `L` are the body field `μ₀`: solid cells drop out
  of the system, and `BC!(μ₀,0,...)` at the walls gives a Neumann pressure condition.
  `measure!(sim)` calls `update!(sim.pois)` after the body moves.
- `solver!` stops when the largest residual is below `tol` and the mean residual below
  `tol/10` (default `tol=2e-3`; `itmx=32` for multigrid). `WaterLily.logger("name")` writes
  the residual history.
- The multigrid coarsens every direction that stays even and larger than 4, and needs at
  least three levels, so domain sizes should be a small integer times a power of two.
- `residual!` zeroes the residual in solid cells and subtracts its mean over the solved
  cells, which keeps the singular Neumann problem solvable.

## Bodies

- An `AbstractBody` must implement `d,n,V = measure(body,x,t;fastd²)`: signed distance,
  normal and velocity. `AutoBody(sdf,map)` gets `n` and `V` by differentiating `sdf` and
  `map`, so motion belongs in the `map`. Use `SA[...]` inside them, never `Array`s.
- `measure!(flow,body)` resets `μ₀`, `μ₁`, `V` over `bbox(body,inside(p))`, computes the
  distance (`measure_sdf!`), fills them over `bbox` again, then applies `BC!`. The force and
  moment integrals in `src/Metrics.jl` also loop over `bbox`.
- `bbox(body,R) = R` by default. A body type that overloads it must return a range that
  also covers every cell the *previous* `measure!` filled, because cells outside it are
  never reset; otherwise a moving body leaves stale `μ₀`/`V` behind.
- The exported name `Bodies` is not defined; set operations on bodies build a `SetBody`.

## Testing

```sh
julia --project -e 'using Pkg; Pkg.test()'                               # CPU, every set
julia --project -e 'using Pkg; Pkg.test(test_args=["poisson","flow"])'   # sets by name prefix
WATERLILY_BACKENDS=cpu,cuda julia --project -e 'using Pkg; Pkg.test(test_args=["--jobs=2"])'
```

- Each `test/test_*.jl` is a set that runs in its own worker. `WATERLILY_BACKENDS`
  (`cpu`, `cuda`, `amdgpu`, `metal`, `all`; default `cpu`) fills the `arrays` list the
  tests loop over; GPU packages are installed on demand. With a GPU, pass a small `--jobs`
  (2 works) or the workers run out of device memory. `WATERLILY_NTHREADS` (default 2) sets
  the threads per worker. The header of `test/runtests.jl` documents the rest.
- The test dependencies are in `test/Project.toml`, which loads WaterLily from this
  checkout. `julia --project=test` gives an environment for running test code by hand.
- With the `"SIMD"` backend preference only `test_alloc.jl` runs (allocation checks);
  with `"KernelAbstractions"` every other set runs. CI runs both, on Julia 1.11 and 1.13,
  and Buildkite runs the GPU backends.
- New behavior needs a test in the matching `test_*.jl`, looping over `arrays` so it also
  runs on GPUs. A change to a hot path must keep `test_alloc.jl` passing.

## Performance

Every contribution comes with benchmark results (see the README): run
[WaterLily-Benchmarks](https://github.com/WaterLily-jl/WaterLily-Benchmarks) against
master, or ask a maintainer to comment `/benchmark` on the PR
(`.github/workflows/benchmark.yml`). Run benchmarks one at a time on an otherwise idle
machine; compare means as well as medians.

## Code style

- Match the surrounding code: terse kernels with few comments. Explain the reasoning in
  the PR description or a docstring, not in long code comments. Public functions get a
  docstring.
- Mutating functions end in `!` and take the mutated argument first.
- Use the existing Unicode names (`ν` is the Greek letter, `μ₀`, `σ`, `ϵ`, `δ`, `∂`).
- Keep the export list short; internal helpers are reached as `WaterLily.name`.
- Extend through dispatch and hooks; never redefine an existing WaterLily method in a
  package.

## Extension points and downstream packages

- `Simulation(...; flow_ctor, pois_ctor)` injects a custom `AbstractFlow` or
  `AbstractPoisson`. `flow_ctor` receives `u0, Δt, ν, g, λ, T, mem, perdir, exitBC`.
- Dispatch on your own subtype: the `mom_step!` phases and `CFL` for an `AbstractFlow`;
  `solver!`, `update!`, `diagonal` (or all of `mom_project!`) for an `AbstractPoisson`;
  `measure`, `measure_sdf!`, `bbox` for an `AbstractBody`; `sim_step!` for an
  `AbstractSimulation`.

These packages use internal functions, so internals are API for them. Search them before
renaming a function or changing its arguments or meaning. The Downstream workflow runs the
tests of the first three, and of ParametricBodies.jl and WaterLilyMeshBodies.jl, on every PR
that changes `src/`, `ext/` or `Project.toml`:

| Package | How it extends WaterLily |
|---------|--------------------------|
| LilyPad.jl | `flow_ctor` with `LilyFlow`; own `mom_predict!`, `mom_correct!`, `CFL` |
| BiotSavartBCs.jl | `pois_ctor` with `BiotSavartPoisson`; own `mom_project!`; uses `div`, `diagonal`, `δv`, `face`, `perBC!` |
| WaterLilyNarrowBand.jl | `NarrowBand <: AbstractBody` with its own `measure_sdf!` and `bbox` |
| WaterLily-Examples | scripts that redefine `mom_step!` and call `scale_u!`, the phases and `CFL` |

A signature change that still dispatches is the dangerous case. For example, a package
method `BDIM!(a::MyFlow)` is no longer called by `mom_correct!`, which calls `BDIM!(a,2)`,
and nothing errors. Keep a cheap compatibility method when it avoids such breaks (as
`scale_u!` is kept, and tested, although `mom_step!` no longer uses it), and describe
changes that affect packages in `NEWS.md`.

## Pull requests

- Keep each PR to one goal, with tests and benchmark results.
- There is no Julia formatter. The Lint workflow runs the pre-commit hooks in
  `.pre-commit-config.yaml` (final newlines, trailing whitespace, Markdown and YAML style)
  and a link checker (lychee, whose exclusions are in the workflow file). Run the hooks
  before pushing: `SKIP=no-commit-to-branch pre-commit run -a` (or
  `pipx run pre-commit run -a` without installing it). They fix most problems in place.
- Note user-visible changes, and changes to internals that packages use, in `NEWS.md`.
- Say in the PR description which parts an AI tool contributed (`CONTRIBUTING.md`).
- Do not commit local settings or output (`Manifest.toml`, `LocalPreferences.toml`, simulation
  files, images, logs).

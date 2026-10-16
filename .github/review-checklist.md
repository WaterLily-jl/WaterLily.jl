### Review checklist

This checklist tells PR authors what reviewers look for, and gives reviewers a structure. The items do not all need to be checked for a PR to be merged.

#### Purpose and scope

- [ ] The PR has a single goal that is clear from its title and description.
- [ ] The changes belong together, and the PR cannot easily be split into smaller ones.

#### Code quality

- [ ] The code is easy to follow and matches the surrounding style: terse kernels, few comments, the existing Unicode names.
- [ ] Grid computations go through `@loop`, work in 2D and 3D, and use one kernel for every backend.
- [ ] There are no leftover debug statements or commented-out code.

#### Documentation

- [ ] New public functions and types have docstrings.
- [ ] User-visible changes, and changes to internal functions that other packages use, are noted in `NEWS.md`, and `AGENTS.md` is updated if it describes them.

#### Testing

- [ ] The tests pass, including the GPU tests on Buildkite.
- [ ] New or changed behavior is tested, looping over `arrays` so the test also runs on GPUs.
- [ ] Hot paths stay allocation-free (`test_alloc.jl`).
- [ ] The downstream tests pass, or the PR says which packages need changes.

#### Performance

- [ ] Benchmark results against master are posted (`/benchmark` comment or WaterLily-Benchmarks).
- [ ] There are no new type instabilities, allocations, or `Float64` values in `Float32` kernels.

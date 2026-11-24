```@raw html
---
layout: home

hero:
  name: WaterLily.jl
  text: Fluid simulation in pure Julia
  tagline: A simple, fast and backend-agnostic solver for incompressible viscous flow around dynamic bodies
  image:
    src: /logo.png
    alt: WaterLily.jl
  actions:
    - theme: brand
      text: Get started
      link: /#getting-started
    - theme: alt
      text: About
      link: /about
    - theme: alt
      text: View on GitHub
      link: https://github.com/WaterLily-jl/WaterLily.jl

features:
  - title: Examples
    details: "Coming soon: tutorial-style examples that guide you through the process of simulating a flow with WaterLily."
    link: /examples/
  - title: API reference
    details: The technical API reference of types, functions and macros (i.e. the documentation strings), organized by the layer of the solver each one belongs to.
    link: /reference/
  - title: Gallery
    details: "Coming soon: some examples of flows simulated with WaterLily and its ecosystem."
    link: /gallery/
  - title: Paper
    details: The 2025 paper in *Computer Physics Communications* describes the main features of the solver and provides benchmarking, validation and profiling results.
    link: https://doi.org/10.1016/j.cpc.2025.109748
  - title: WaterLily-Examples
    details: A much larger collection of example programs and notebooks, including ones that need packages WaterLily itself does not depend on, such as parametric bodies, Biot-Savart boundary conditions and optimisation.
    link: https://github.com/WaterLily-jl/WaterLily-Examples
  - title: JuliaCon2024 talk
    details: A 25-minute introduction to the solver and what it is for.
    link: https://www.youtube.com/watch?v=FwMh2rq9kOU
---
```

## Getting started

As a new user of WaterLily it is suggested to start working with examples before using WaterLily to tackle the specific flow you ultimately want to simulate. The [README](https://github.com/WaterLily-jl/WaterLily.jl#readme) walks through the flow over a circle, which already introduces the three pieces every simulation is built from -- an implicitly defined geometry, a `Simulation`, and a time loop you measure from. The [WaterLily-Examples](https://github.com/WaterLily-jl/WaterLily-Examples) repository then holds a much larger collection of programs and notebooks. Tutorial-style examples will be added to these pages soon.

### Installation

To use WaterLily you first need to install Julia, see <https://julialang.org/> for details. Installing WaterLily can then be done from the Pkg REPL; press `]` at the `julia>` prompt to enter `pkg>` mode:

```julia
pkg> add WaterLily
```

This will install WaterLily and all necessary dependencies. Press backspace to get back to the `julia>` prompt. (See the [documentation for Pkg](https://pkgdocs.julialang.org/), Julia's package manager, for more help regarding package installation and project management.)

WaterLily is written entirely in Julia and depends only on registered Julia packages, so it runs on Linux, macOS and Windows. Some capabilities live in package extensions and switch on when you load the relevant package, none of which is required to run a simulation:

```julia
pkg> add Plots      # the plotting helpers
pkg> add CUDA       # to run on an NVIDIA GPU (or AMDGPU, for a ROCm device)
pkg> add WriteVTK   # to write ParaView output
```

Finally, to load WaterLily, use

```julia
using WaterLily
```

Note that multi-threading on the CPU requires *starting* Julia with the `--threads` argument -- see the [multi-threading section](https://docs.julialang.org/en/v1/manual/multi-threading/) of the Julia manual. If you start Julia with multiple threads, WaterLily detects this and multi-threads its loops automatically.

You are now all set to start using WaterLily!

## Contributing to WaterLily

We always appreciate new contributions, so please [submit a pull request](https://github.com/WaterLily-jl/WaterLily.jl/compare) with your changes and help us make WaterLily even better! Note that contributions need to be submitted together with benchmark results - WaterLily should always be fast! 😃 For this, we have a [fully automated benchmarking suite](https://github.com/WaterLily-jl/WaterLily-Benchmarks) that conducts performance tests. In short, to compare your changes with the latest WaterLily, clone that repo and run the benchmarks with

``` sh
git clone https://github.com/WaterLily-jl/WaterLily-Benchmarks && cd WaterLily-Benchmarks
sh benchmark.sh -wd "<your/waterlily/path>" -w "<your_waterlily_branch> main"
julia --project compare.jl
```

This will run benchmarks for CPU and GPU backends. If you do not have a GPU, simply pass `-b "Array"` when running `benchmark.sh`. More information on the benchmark suite is available in that [README](https://github.com/WaterLily-jl/WaterLily-Benchmarks/blob/main/README.md).

Of course, ideas, suggestions, and questions are welcome too! Please [raise an issue](https://github.com/WaterLily-jl/WaterLily.jl/issues/new/choose) to address any of these.

### Getting help

If you have questions about WaterLily, or ideas and suggestions for it, you are encouraged to [raise an issue](https://github.com/WaterLily-jl/WaterLily.jl/issues/new/choose) on the GitHub repository. Bug reports are most useful when they come with a short program that reproduces the problem.

## Citing WaterLily

If you have used WaterLily for research, please **cite us**! The [2025 paper](https://doi.org/10.1016/j.cpc.2025.109748) in *Computer Physics Communications* describes the main features of the solver and provides benchmarking, validation and profiling results.

```tex
@article{WeymouthFont2025,
    author  = {G.D. Weymouth and B. Font},
    title   = {WaterLily.jl: A differentiable and backend-agnostic Julia solver
                for incompressible viscous flow around dynamic bodies},
    doi     = {10.1016/j.cpc.2025.109748},
    journal = {Computer Physics Communications},
    year    = {2025},
    volume  = {315},
    pages   = {109748},
}
```

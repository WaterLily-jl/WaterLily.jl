```@meta
CurrentModule = WaterLily
```

# Simulation

A [`Simulation`](@ref) bundles everything a run needs (the grid, the [flow fields](flow.md),
the [immersed geometry](bodies.md) and the [pressure solver](poisson.md)), and
[`sim_step!`](@ref) advances it. For most users this page is the whole interface: build a
`Simulation`, step it, and [measure](metrics.md) whatever you are after between steps.

See [Flow over a circle](@ref examples-TwoD_Circle) for the shortest complete example.

```@autodocs
Modules = [WaterLily]
Pages   = ["WaterLily.jl"]
Order   = [:type, :function, :macro, :constant]
```

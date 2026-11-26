```@meta
CurrentModule = WaterLily
```

# Pressure Poisson solvers

Every time step ends by projecting the velocity onto a divergence-free field, which means
solving a variable-coefficient Poisson equation for the pressure. This is the expensive part of
an incompressible solve, and the reason WaterLily is fast.

[`Poisson`](@ref) is the operator, together with its smoothers and a preconditioned conjugate
gradient solver. [`MultiLevelPoisson`](@ref) wraps it in a geometric multigrid V-cycle,
restricting the problem onto progressively coarser grids so that the cost stays near-linear in
the number of cells; it is what a [`Simulation`](@ref) uses by default. Pass `pois_ctor` to
`Simulation` to substitute your own solver.

The convergence of each solve can be recorded with [`logger`](@ref) and inspected afterwards with
`plot_logger`.

## Operators

```@autodocs
Modules = [WaterLily]
Pages   = ["Poisson.jl"] # also matches MultiLevelPoisson.jl
Order   = [:type]
```

## Smoothers and solvers

```@autodocs
Modules = [WaterLily]
Pages   = ["Poisson.jl"] # also matches MultiLevelPoisson.jl
Order   = [:function, :macro, :constant]
```

```@meta
CurrentModule = WaterLily
```

# Grid, loops and boundary conditions

WaterLily works on a uniform Cartesian grid with one layer of ghost cells on every face.
Scalars such as the pressure live at cell centres and are indexed `p[I]`; vectors such as the
velocity live on cell *faces* and are indexed `u[I, i]`, where `u[I, i]` is the `i`-component on
the lower face of cell `I` in direction `i`. [`loc`](@ref) gives the physical position of either,
and [`inside`](@ref) gives the range of real (non-ghost) cells.

Everything in the solver is written with [`@loop`](@ref) and [`@inside`](@ref), which expand to
KernelAbstractions kernels. That is what makes the same source run serially, multi-threaded, or
on a GPU. Write your own metrics and body functions the same way and they will too.

[`BC!`](@ref) applies the domain boundary conditions each stage: a Dirichlet condition on the
normal velocity and a Neumann condition on the tangential components, with periodic directions
and a convective exit handled by [`perBC!`](@ref) and [`exitBC!`](@ref).

## Grid conventions and indexing

```@autodocs
Modules = [WaterLily]
Pages   = ["core.jl"]
Filter  = f -> nameof(f) in (:loc, :inside, :inside_u, :slice, :face, :δ, :CIj)
Order   = [:function]
```

## Loop macros

```@autodocs
Modules = [WaterLily]
Pages   = ["core.jl"]
Order   = [:macro]
```

## Boundary conditions

```@autodocs
Modules = [WaterLily]
Pages   = ["core.jl"]
Filter  = f -> nameof(f) in (:BC!, :perBC!, :exitBC!)
Order   = [:function]
```

## Backends

The kernel launch configuration, and arrays allocated on the backend of a simulation.

```@autodocs
Modules = [WaterLily]
Pages   = ["core.jl"]
Filter  = f -> nameof(f) in (:workgroup, :zeros_on)
Order   = [:function]
```

## Logging

[`logger`](@ref) records the residuals and iteration counts of every pressure solve to a file.

```@autodocs
Modules = [WaterLily]
Pages   = ["core.jl"]
Filter  = f -> !(f isa Function) || !(nameof(f) in (:loc, :inside, :inside_u, :slice, :face,
    :δ, :CIj, :BC!, :perBC!, :exitBC!, :workgroup, :zeros_on))
Order   = [:type, :function, :constant]
```

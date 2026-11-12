```@meta
CurrentModule = WaterLily
```

# Bodies and geometry

Geometry in WaterLily is *implicit*: a body is a signed distance function, optionally composed
with a coordinate `map` that places or moves it. [`AutoBody`](@ref) wraps the two and uses
automatic differentiation to derive the surface normal and the body velocity, so neither has to
be supplied by hand. Bodies combine with the set operations `∪`, `∩` and `-` into a lazy
[`SetBody`](@ref) tree, which is how composite geometries are built.

`measure!` is what connects a body to the grid: it fills the Boundary Data Immersion Method
kernel moments that the [momentum step](flow.md) uses to enforce the no-slip condition.

Worked examples: [a square from four planes](@ref examples-TwoD_Square) (set operations and
maps), [many bodies at once](@ref examples-TwoD_MultipleBodies) (unions) and
[a hovering plate](@ref examples-TwoD_Hover) (a time-dependent map).

## Body types

```@autodocs
Modules = [WaterLily]
Pages   = ["Body.jl", "RigidMap.jl"] # "Body.jl" also matches AutoBody.jl
Order   = [:type]
```

## Measuring a body

```@autodocs
Modules = [WaterLily]
Pages   = ["Body.jl", "RigidMap.jl"] # "Body.jl" also matches AutoBody.jl
Order   = [:function, :macro, :constant]
```

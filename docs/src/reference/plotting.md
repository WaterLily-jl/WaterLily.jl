```@meta
CurrentModule = WaterLily
```

# Plotting

WaterLily has two sets of plotting helpers, each in a package extension that only exists once
the corresponding plotting package is loaded.

## Plots

`using Plots` adds a few thin wrappers. `flood` is a filled contour plot with the aspect ratio
and colour limits a flow field usually wants, and `sim_gif!` is the step-plot-repeat loop from
the examples bundled into one call. Most examples on these pages use them.

```@autodocs
Modules = [Base.get_extension(WaterLily, :WaterLilyPlotsExt)]
Order   = [:function, :macro, :constant]
```

## Makie

For interactive and 3D rendering (volumes, iso-surfaces, pathlines and videos), load a Makie
backend instead: CairoMakie to write images and videos, or GLMakie for an interactive window and
GPU volume rendering. [`viz!`](@ref) then steps the simulation and renders it in one call, and
[`viz_step!`](@ref) drives the same figure from a loop of your own. The
[flow over a sphere](@ref examples-ThreeD_Sphere) example uses it.

Rendering the body as a mesh (`body2mesh=true`) or a field as an iso-surface (`isomesh`) also
needs `using Meshing`, which activates a second extension.

```@docs
viz!
viz_step!
get_body
plot_body_obs!
```

The helpers below are internal to the extension, and documented here for anyone extending it.

```@autodocs
Modules = [Base.get_extension(WaterLily, :WaterLilyMakieExt)]
Filter  = f -> !(nameof(f) in (:viz!, :viz_step!, :get_body, :plot_body_obs!))
Order   = [:function, :macro, :constant]
```

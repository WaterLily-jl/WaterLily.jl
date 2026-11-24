# WaterLily.jl

WaterLily is a simple and fast fluid simulator written in pure [Julia](https://julialang.org/). It solves the unsteady incompressible Navier-Stokes equations on a Cartesian grid, immerses solid boundaries using the Boundary Data Immersion Method, and runs on serial CPU, multi-threaded CPU, and GPU backends. The aim is to be i) simple to read and extend, ii) fast, and iii) backend-agnostic.

## Capabilities
 - Incompressible viscous flow in two and three dimensions, around bodies that can move and deform. A body is defined by a signed distance function and an optional coordinate map, so there is no mesh to generate or update ([bodies and geometry](reference/bodies.md)).
 - Composite geometries built from simpler ones with set operations, and forces and moments measured on the whole body or on any part of it ([flow metrics](reference/metrics.md)).
 - Uniform, space- and time-varying inflow, periodic directions, a convective exit, and a body force through the domain acceleration `g(i,x,t)` ([simulation](reference/simulation.md)).
 - The same source runs serially, multi-threaded, or on a GPU, by passing for example `mem=CuArray` (NVIDIA) or `mem=ROCArray` (AMD) ([grid, loops and boundary conditions](reference/grid.md)).
 - The solver is differentiable: it runs on ForwardDiff dual numbers, so a simulation can be differentiated with respect to its parameters.
 - Time averaging, sub-grid-scale modelling, VTK output for ParaView, JLD2 restarts, and plotting with Plots or Makie ([input and output](reference/io.md), [plotting](reference/plotting.md), [utilities](reference/utilities.md)).

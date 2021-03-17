# WaterLily.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/gabrielweymouth/WaterLily.jl.svg?branch=master)](https://travis-ci.com/gabrielweymouth/WaterLily.jl)
[![codecov.io](http://codecov.io/github/gabrielweymouth/WaterLily.jl/coverage.svg?branch=master)](http://codecov.io/github/gabrielweymouth/WaterLily.jl?branch=master)
<!--
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://gabrielweymouth.github.io/WaterLily.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://gabrielweymouth.github.io/WaterLily.jl/dev)
-->

![Julia flow](examples/julia.gif)

## Overview

WaterLily.jl is a real-time fluid simulator written in pure Julia. This is an experimental project to take advantage of the active scientific community in Julia to accelerate and enhance fluid simulations. If you want to play around with a much more fully developed and documented solver right now, you should head over to [LilyPad](https://github.com/weymouth/lily-pad).

## Method/capabilities

WaterLily.jl solves the unsteady incompressible 2D or 3D [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) on a Cartesian grid. The pressure Poisson equation is solved with a [geometric multigrid](https://en.wikipedia.org/wiki/Multigrid_method) method. Solid boundaries are modelled using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/).

## Examples

The user can set the boundary conditions, the initial velocity field, the fluid viscosity (which determines the [Reynolds number](https://en.wikipedia.org/wiki/Reynolds_number)), and immerse solid obstacles using a signed distance function. These examples and others are found in the [examples](examples).

### Flow over a circle
We define the size of the simulation domain as `n`x`m` cells. The circle has radius `R=m/8` and is centered at `[m/2,m/2]`. The flow boundary conditions are `[U=1,0]` and Reynolds number is `Re=UR/ν`. 
```julia
using WaterLily
using LinearAlgebra: norm2

function circle(n,m;Re=250)
    # Set physical parameters
    U,R,center = 1., m/8., [m/2,m/2]
    ν=U*R/Re
    @show R,ν

    body = AutoBody((x,t)->norm2(x .- center) - R)
    Simulation((n+2,m+2), [U,0.], R; ν, body)
end
```
The second to last line defines the circle geometry using a signed distance function. The `AutoBody` function uses [automatic differentiation](https://github.com/JuliaDiff/) to infer the other geometric parameter automatically. Replace the circle's distance function with any other, and now you have the flow around something else... such as a [donut](ThreeD_donut.jl), a [block](TwoD_block.jl) or the [Julia logo](TwoD_Julia.jl). Finally, the last line defines the `Simulation` by passing in the `dims=(n+2,m+2)` and the other parameters we've defined.

Now we can create a simulation (first line) and run it forward in time (second line)
```julia
circ = circle(3*2^6,2^7);
sim_step!(circ,t_end=10)
```
Note we've set `n,m` to be multiples of powers of 2, which is important when using the (very fast) Multi-Grid solver. We can now access and plot whatever variables we like. For example, we could print the velocity at `I::CartesianIndex` using `println(sim.flow.u[I])` or plot the whole pressure field using
```julia
using Plots
contour(sim.flow.p')
```
A set of [flow metric functions](src/Metrics.jl) have been implemented and the examples use these to make gifs such as the one above.

### 3D Taylor Green Vortex
You can also simulate a nontrivial initial velocity field by passing in a vector function.
```julia
function TGV(p=6,Re=1e5)
    # Define vortex size, velocity, viscosity
    L = 2^p; U = 1; ν = U*L/Re

    function uλ(i,vx) # vector function
        x,y,z = @. (vx-1.5)*π/L                # scaled coordinates
        i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x
        i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y
        return 0.                              # u_z
    end

    # Initialize simulation
    Simulation((L+2,L+2,L+2), zeros(3), L; uλ, ν, U)
end
```
The velocity field is defined by the vector component `i` and the 3D position vector `vx`. We scale the coordinates so the velocity will be zero on the domain boundaries and then check which component is needed and return the correct expression.

## Development goals
 - Immerse obstacles defined by 3D meshes or 2D lines using [GeometryBasics](https://github.com/JuliaGeometry/GeometryBasics.jl).
 - GPU acceleration with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).
 - Split multigrid method into its own repository, possibly merging with [AlgebraicMultigrid](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) or [IterativeSolvers](https://github.com/JuliaMath/IterativeSolvers.jl).

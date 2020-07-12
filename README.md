# WaterLily

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

Real-time fluid simulator in Julia.


## Overview

WaterLily is an experimental Julia port of [LilyPad](https://github.com/weymouth/lily-pad). The motivation for the port was to take advantage of the larger scientific community in Julia (compared to Processing), but if you want to play around with a much more fully developed solver right now, you should head over to [LilyPad](https://github.com/weymouth/lily-pad).

## Method/capabilities

WaterLily solves the unsteady incompressible 2D or 3D [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) on a Cartesian grid. The pressure Poisson equation is solved with a [geometric multigrid](https://en.wikipedia.org/wiki/Multigrid_method) method.

## Development goals
 - Full implementation of the [Boundary Data Immersion Method](https://www.sciencedirect.com/science/article/pii/S0021999116307148), probably using STLs to define solid geometries. 
 - GPU acceleration with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
 - Split multigrid method into its own repository, possibly merging with [AlgebraicMultigrid](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) or [IterativeSolvers](https://github.com/JuliaMath/IterativeSolvers.jl).
 - Optimize for [autodiff](https://github.com/JuliaDiff/) 

## WaterLily.jl

WaterLily is a simple and fast fluid simulator written in pure [Julia](https://julialang.org/). It solves the unsteady incompressible Navier-Stokes equations on a Cartesian grid, immerses solid boundaries using the Boundary Data Immersion Method, and runs on serial CPU, multi-threaded CPU, and GPU backends. The aim is to be i) simple to read and extend, ii) fast, and iii) backend-agnostic.

## Capabilities
 - WaterLily can simulate incompressible viscous flow around dynamic bodies in two and three dimensions, with or without free surfaces.

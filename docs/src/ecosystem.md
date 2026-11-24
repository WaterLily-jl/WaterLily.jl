## The wider ecosystem

The WaterLily ecosystem is growing, with a number of packages that extend the core functionality of WaterLily. These include:

- [**LilyPad.jl**](https://github.com/WaterLily-jl/LilyPad.jl) -- a semi-Lagrangian momentum-step variant of WaterLily for real-time simulations.
- [**ParametricBodies.jl**](https://github.com/WaterLily-jl/ParametricBodies.jl) -- bodies defined by parametric curves and NURBS rather than by a distance function, including bodies that deform while the simulation runs.
  Repositories for geometry, boundary conditions, and visualization
- [**WaterLilyMeshBodies.jl**](https://github.com/WaterLily-jl/WaterLilyMeshBodies.jl) -- tools to import mesh-defined geometries into WaterLily.
- [**BiotSavartBCs.jl**](https://github.com/WaterLily-jl/BiotSavartBCs.jl) -- replaces the domain boundary condition with a Biot-Savart evaluation of the velocity induced by the vorticity in the domain, so a much smaller domain behaves as if it were unbounded. Useful for freely moving bodies and for forces that are sensitive to blockage.
- [**Pathlines.jl**](https://github.com/WaterLily-jl/Pathlines.jl) -- Pathline visualization tools for WaterLily simulations, allows to track the motion of fluid particles over time.
- [**WaterLilyNarrowBand.jl**](https://github.com/WaterLily-jl/WaterLilyNarrowBand.jl) -- Measure moving bodies faster using a narrow band around their surface.
- [**CoupledSimulations.jl**](https://github.com/WaterLily-jl/CoupledSimulations.jl) -- Couples WaterLily with other simulation codes, such as structural solvers, to allow for fluid-structure interaction simulations.
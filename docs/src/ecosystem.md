## The wider ecosystem

These are the packages a gallery program is most likely to reach for:

- [**ParametricBodies.jl**](https://github.com/WaterLily-jl/ParametricBodies.jl) -- bodies defined
  by parametric curves and NURBS rather than by a distance function, including bodies that deform
  while the simulation runs.
- [**BiotSavartBCs.jl**](https://github.com/WaterLily-jl/BiotSavartBCs.jl) -- replaces the domain
  boundary condition with a Biot-Savart evaluation of the velocity induced by the vorticity in the
  domain, so a much smaller domain behaves as if it were unbounded. Useful for freely moving bodies
  and for forces that are sensitive to blockage.
- [**ForcePartition.jl**](https://github.com/WaterLily-jl/ForcePartition.jl) -- splits a measured
  force into the mechanisms that produce it (vortex-induced, added-mass, viscous) rather than
  reporting a single number.
- [**WaterLilyPreCICE.jl**](https://github.com/WaterLily-jl/WaterLilyPreCICE.jl) -- couples
  WaterLily to a structural solver through [preCICE](https://precice.org), for fluid-structure
  interaction where the structure needs a code of its own.

Of these, only ParametricBodies is in the General registry; the others are installed from their
repository URL. Since gallery programs are not executed here, that makes no difference to the
documentation build -- it only matters when you run one yourself.
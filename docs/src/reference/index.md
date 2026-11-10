# [API reference](@id api-reference)

- **[Simulation](simulation.md)**: the top-level object and its time loop. This is the
  entire public interface for most users.
- **[Bodies and geometry](bodies.md)**: immersed geometry, that is signed distance functions,
  coordinate maps, set operations, and how the solver measures a body onto the grid.
- **[Flow and time stepping](flow.md)**: the velocity and pressure fields, and the
  fractional-step momentum integrator that advances them.
- **[Pressure Poisson solvers](poisson.md)**: the projection step, with the Poisson operator and
  the geometric multigrid solver that inverts it.
- **[Flow metrics](metrics.md)**: measuring the flow, from forces and moments on a body to
  vorticity and other velocity-gradient quantities, and time averaging.
- **[Grid, loops and boundary conditions](grid.md)**: the Cartesian grid conventions, the
  loop macros that every kernel is written with, and the domain boundary conditions.
- **[Input and output](io.md)**: writing and restarting simulations (ParaView and JLD2).
- **[Plotting](plotting.md)**: the Plots helpers used throughout the examples, and `viz!`
  for interactive and 3D rendering with Makie.
- **[Utilities](utilities.md)**: interpolation, sub-grid-scale modelling, and 2D-to-3D
  field spreading.

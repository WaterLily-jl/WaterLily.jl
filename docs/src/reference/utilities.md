```@meta
CurrentModule = WaterLily
```

# Utilities

Supporting functions that do not belong to one layer of the solver.

- `interp` samples a grid field at an arbitrary position, which is what you need for probes,
  pathlines, or coupling to another solver.
- `sgs!` adds a sub-grid-scale eddy viscosity, for running under-resolved at high Reynolds
  number.
- `spread!` extrudes a 2D field into a 3D one, so a cheap 2D run can initialise a 3D one.

```@autodocs
Modules = [WaterLily]
Pages   = ["util.jl"]
Order   = [:type, :function, :macro, :constant]
```

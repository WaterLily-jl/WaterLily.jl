```@meta
CurrentModule = WaterLily
```

# Input and output

Simulation output is handled by package extensions, so these functions only exist once the
corresponding package is loaded.

- **ParaView** (`using WriteVTK`): `vtkWriter` opens a `.pvd` collection and `save!` appends the
  current state as a time step, giving a file ParaView can animate. Which fields are written is
  controlled by the `attrib` dictionary (see `default_attrib`). `using ReadVTK` adds the matching
  `load!`, for restarting from a written collection.
- **Checkpoints** (`using JLD2`): `save!` and `load!` on a `Flow` or a `MeanFlow` write and read
  the raw arrays, which is the cheaper way to restart a run that you do not need to visualise.

## ParaView output

```@autodocs
Modules = [Base.get_extension(WaterLily, :WaterLilyWriteVTKExt)]
Order   = [:type, :function, :macro, :constant]
```

## Restarting from ParaView output

```@autodocs
Modules = [Base.get_extension(WaterLily, :WaterLilyReadVTKExt)]
Order   = [:type, :function, :macro, :constant]
```

## JLD2 checkpoints

```@autodocs
Modules = [Base.get_extension(WaterLily, :WaterLilyJLD2Ext)]
Order   = [:type, :function, :macro, :constant]
```

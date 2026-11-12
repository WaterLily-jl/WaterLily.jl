```@meta
CurrentModule = WaterLily
```

# Flow and time stepping

A [`Flow`](@ref) holds the primary variables (the face-centred velocity `u` and the
cell-centred pressure `p`) along with the immersion kernel moments measured from the
[body](bodies.md) and the time-step history `Δt`.

[`mom_step!`](@ref) is the integrator: a two-stage fractional step, each stage computing
convection and diffusion, applying the Boundary Data Immersion Method forcing, and then
projecting the velocity onto a divergence-free field with the
[pressure solver](poisson.md). `mom_predict!`, `mom_correct!` and `mom_project!` are its
pieces, exposed so that a custom time integrator can reuse them.

```@autodocs
Modules = [WaterLily]
Pages   = ["Flow.jl"]
Order   = [:type, :function, :macro, :constant]
```

module WaterLily

include("util.jl")
export L₂,BC!,@inside,inside,δ

include("Poisson.jl")
export AbstractPoisson,Poisson,solve!,mult

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solve!,mult

include("Body.jl")
export AbstractBody,BDIM_coef,apply

include("AutoBody.jl")
export AutoBody,measure

include("Flow.jl")
export Flow,mom_step!

include("Metrics.jl")

struct Simulation
    U :: Number # velocity scale
    L :: Number # length scale
    flow :: Flow
    pois :: AbstractPoisson
end
"""
    sim_time(sim::Simulation)

Return the current dimensionless time of the simulation `tU/L`
where `t=sum(Δt)`, and `U`,`L` are the simulation velocity and length
scales.
"""
sim_time(sim::Simulation) = sum(sim.flow.Δt)*sim.U/sim.L
"""
    sim_step!(sim::Simulation,t_end;verbose=false)

Integrate the simulation `sim` up to dimensionless time `t_end`.
If `verbose=true` the time `tU/L` and adaptive time step `Δt` are
printed every time step.
"""
function sim_step!(sim::Simulation,t_end;verbose=false)
    t = sim_time(sim)
    while t < t_end
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]*sim.U/sim.L
        verbose && println("tU/L=",round(t,digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

export Simulation,sim_step!,sim_time
end # module

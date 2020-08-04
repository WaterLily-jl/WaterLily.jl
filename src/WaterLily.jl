module WaterLily

include("util.jl")
export L₂,BC!,@inside,inside,δ

include("Poisson.jl")
export AbstractPoisson,Poisson,solve!,mult

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solve!,mult

include("Body.jl")
export BDIM_coef,apply

include("Flow.jl")
export Flow,mom_step!

include("Metrics.jl")

struct Simulation
    U :: Number # velocity scale
    L :: Number # length scale
    a :: Flow
    b :: AbstractPoisson
end

sim_time(sim) = sum(sim.a.Δt*sim.U/sim.L)

function sim_step!(sim::Simulation,t_end;verbose=false)
    t = sim_time(sim)
    t_0 = t
    while t < t_end
        mom_step!(sim.a,sim.b) # evolve Flow
        t += sim.a.Δt[end]*sim.U/sim.L
        verbose && println("tU/L=",round(t,digits=4),
            ", Δt=",round(sim.a.Δt[end],digits=3))
    end
end

export Simulation,sim_step!
end # module

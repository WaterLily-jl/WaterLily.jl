module WaterLily

include("util.jl")
export L₂,BC!,@inside,inside,δ,apply,apply!

include("Poisson.jl")
export AbstractPoisson,Poisson,solver!,mult

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solver!,mult

include("Flow.jl")
export Flow,mom_step!

include("Body.jl")
export AbstractBody,measure!

include("AutoBody.jl")
export AutoBody,measure

include("Metrics.jl")
using LinearAlgebra: norm2

"""
    Simulation(dims::Tuple, u_BC::Vector, L::Number;
               U=norm2(u_BC), Δt=0.25, ν=0.,
               uλ::Function=(i,x)->u_BC[i],
               body::AbstractBody=NoBody())

Constructor for a WaterLily.jl simulation:
    - `dims`: Simulation domain dimensions.
    - `u_BC`: Simulation domain velocity boundary conditions, `u_BC[i]=uᵢ, i=1,2...`.
    - `L`: Simulation length scale.
    - `U`: Simulation velocity scale.
    - `Δt`: Initial time step.
    - `ν`: Scaled viscosity (`Re=UL/ν`)
    - `uλ`: Function to generate the initial velocity field.
    - `body`: Embedded geometry

See files in `examples` folder for examples.
"""
struct Simulation
    U :: Number # velocity scale
    L :: Number # length scale
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    function Simulation(dims::Tuple, u_BC::Vector, L::Number;
                        Δt=0.25, ν=0., U=norm2(u_BC),
                        uλ::Function=(i,x)->u_BC[i],
                        body::AbstractBody=NoBody())
        flow = Flow(dims,u_BC;uλ,Δt,ν)
        measure!(flow,body)
        new(U,L,flow,body,MultiLevelPoisson(flow.μ₀))
    end
end

time(sim::Simulation) = sum(sim.flow.Δt[1:end-1])
"""
    sim_time(sim::Simulation)

Return the current dimensionless time of the simulation `tU/L`
where `t=sum(Δt)`, and `U`,`L` are the simulation velocity and length
scales.
"""
sim_time(sim::Simulation) = time(sim)*sim.U/sim.L

"""
    sim_step!(sim::Simulation,t_end;verbose=false)

Integrate the simulation `sim` up to dimensionless time `t_end`.
If `verbose=true` the time `tU/L` and adaptive time step `Δt` are
printed every time step.
"""
function sim_step!(sim::Simulation,t_end;verbose=false,remeasure=false)
    t = time(sim)
    while t < t_end*sim.L/sim.U
        remeasure && measure!(sim,t)
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
        verbose && println("tU/L=",round(t*sim.U/sim.L,digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

"""
    measure!(sim::Simulation,t=time(sim))

Measure a dynamic `body` to update the `flow` and `pois` coefficients.
"""
function measure!(sim::Simulation,t=time(sim))
    measure!(sim.flow,sim.body;t)
    update!(sim.pois,sim.flow.μ₀)
end

export Simulation,sim_step!,sim_time,measure!
end # module

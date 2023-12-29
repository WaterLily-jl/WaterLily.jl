module WaterLily

include("util.jl")
export L₂,BC!,@inside,inside,δ,apply!,loc

using Reexport
@reexport using KernelAbstractions: @kernel,@index,get_backend

include("Poisson.jl")
export AbstractPoisson,Poisson,solver!,mult!

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solver!,mult!

include("Flow.jl")
export Flow,mom_step!

include("Body.jl")
export AbstractBody,measure_sdf!

include("AutoBody.jl")
export AutoBody,measure,sdf,+,-

include("Metrics.jl")

"""
    Simulation(dims::NTuple, u_BC::NTuple, L::Number;
               U=norm2(u_BC), Δt=0.25, ν=0., ϵ=1,
               uλ::Function=(i,x)->u_BC[i],
               body::AbstractBody=NoBody(),
               T=Float32, mem=Array)

Constructor for a WaterLily.jl simulation:

  - `dims`: Simulation domain dimensions.
  - `u_BC`: Simulation domain velocity boundary conditions, `u_BC[i]=uᵢ, i=eachindex(dims)`.
  - `L`: Simulation length scale.
  - `U`: Simulation velocity scale.
  - `Δt`: Initial time step.
  - `ν`: Scaled viscosity (`Re=UL/ν`).
  - `ϵ`: BDIM kernel width.
  - `uλ`: Function to generate the initial velocity field.
  - `body`: Immersed geometry.
  - `T`: Array element type.
  - `mem`: memory location. `Array` and `CuArray` run on CPU and CUDA backends, respectively.

See files in `examples` folder for examples.
"""
struct Simulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    function Simulation(dims::NTuple{N}, u_BC::NTuple{N}, L::Number;
                        Δt=0.25, ν=0., g=(i,t)->0, U=√sum(abs2,u_BC), ϵ=1, perdir=(0,),
                        uλ::Function=(i,x)->u_BC[i], exitBC=false,
                        body::AbstractBody=NoBody(),T=Float32,mem=Array) where N
        flow = Flow(dims,u_BC;uλ,Δt,ν,g,T,f=mem,perdir,exitBC)
        measure!(flow,body;ϵ)
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir))
    end
end

time(flow::Flow) = sum(flow.Δt[1:end-1])
time(sim::Simulation) = time(sim.flow)
timeNext(flow::Flow) = sum(flow.Δt)
"""
    sim_time(sim::Simulation)

Return the current dimensionless time of the simulation `tU/L`
where `t=sum(Δt)`, and `U`,`L` are the simulation velocity and length
scales.
"""
sim_time(sim::Simulation) = time(sim)*sim.U/sim.L

"""
    sim_step!(sim::Simulation,t_end;max_steps=typemax(Int),remeasure=true,verbose=false)

Integrate the simulation `sim` up to dimensionless time `t_end`.
If `remeasure=true`, the body is remeasured at every time step.
Can be set to `false` for static geometries to speed up simulation.
"""
function sim_step!(sim::Simulation,t_end;max_steps=typemax(Int),verbose=false,remeasure=true)
    t = time(sim)
    while t < t_end*sim.L/sim.U && length(sim.flow.Δt) <= max_steps
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
    measure!(sim.flow,sim.body;t,ϵ=sim.ϵ)
    update!(sim.pois)
end

export Simulation,sim_step!,sim_time,measure!

include("vtkWriter.jl")
export vtkWriter,write!,close

end # module

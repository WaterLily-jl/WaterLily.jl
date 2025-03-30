"""
$(README)
"""
module WaterLily

using DocStringExtensions

include("util.jl")
export L₂,BC!,@inside,inside,δ,apply!,loc,@log

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
export AutoBody,Bodies,measure,sdf,+,-

include("Metrics.jl")

abstract type AbstractSimulation end
"""
    Simulation(dims::NTuple, Uλ::Union{NTuple,Function}, L::Number;
               U=norm2(Uλ), Δt=0.25, ν=0., ϵ=1, g=nothing, 
               perdir=(), exitBC=false,
               body::AbstractBody=NoBody(),
               T=Float32, mem=Array)

Constructor for a WaterLily.jl simulation:

  - `dims`: Simulation domain dimensions.
  - `Uλ`: Domain velocity field applied to the initial, boundary and acceleration conditions.
        Define a uniform field by `Uλ[i]::NTuple`, or supply a function of space and time `Uλ(i,x,t)`.
  - `L`: Simulation length scale.
  - `U`: Simulation velocity scale. Required if using `Uλ::Function`.
  - `Δt`: Initial time step.
  - `ν`: Scaled viscosity (`Re=UL/ν`).
  - `g`: Domain acceleration, `g(i,x,t)=duᵢ/dt`
  - `ϵ`: BDIM kernel width.
  - `perdir`: Domain periodic boundary condition in the `(i,)` direction.
  - `exitBC`: Convective exit boundary condition in the `i=1` direction.
  - `body`: Immersed geometry.
  - `T`: Array element type.
  - `mem`: memory location. `Array`, `CuArray`, `ROCm` to run on CPU, NVIDIA, or AMD devices, respectively.

See files in `examples` folder for examples.
"""
mutable struct Simulation <: AbstractSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    function Simulation(dims::NTuple{N}, Uλ, L::Number;
                        Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, perdir=(),
                        exitBC=false, body::AbstractBody=NoBody(),
                        T=Float32, mem=Array) where N
        @assert !(isnothing(U) && isa(Uλ,Function)) "`U` must be specified if `u_BC` is a Function"
        isnothing(U) && (U = √sum(abs2,Uλ))
        check_fn(g,N,T); check_fn(Uλ,N,T)
        flow = Flow(dims,Uλ;Δt,ν,g,T,f=mem,perdir,exitBC)
        measure!(flow,body;ϵ)
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir))
    end
end
function check_fn(f::Function,N,T)
    @assert first(methods(f)).nargs==4 "$f needs to be defined as f(i,x,t)"
    @assert all(typeof.(ntuple(i->f(i,zeros(SVector{N,T}),zero(T)),N)).==T) "$f is not type stable"
end
check_fn(f,N,T) = nothing

time(sim::AbstractSimulation) = time(sim.flow)
"""
    sim_time(sim::Simulation)

Return the current dimensionless time of the simulation `tU/L`
where `t=sum(Δt)`, and `U`,`L` are the simulation velocity and length
scales.
"""
sim_time(sim::AbstractSimulation) = time(sim)*sim.U/sim.L

"""
    sim_step!(sim::Simulation,t_end=sim(time)+Δt;max_steps=typemax(Int),remeasure=true,verbose=false)

Integrate the simulation `sim` up to dimensionless time `t_end`.
If `remeasure=true`, the body is remeasured at every time step.
Can be set to `false` for static geometries to speed up simulation.
A user-defined function `udf` can be passed to arbitrarily modify the `::Flow` during the predictor and corrector steps.
If the `udf` user keyword arguments, these needs to be included in the `sim_step!` call as well.
"""
function sim_step!(sim::AbstractSimulation,t_end;remeasure=true,max_steps=typemax(Int),udf=nothing,verbose=false,kwargs...)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure, udf, kwargs...)
        verbose && println("tU/L=",round(sim_time(sim),digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end
function sim_step!(sim::AbstractSimulation;remeasure=true,udf=nothing,kwargs...)
    remeasure && measure!(sim)
    mom_step!(sim.flow, sim.pois; udf, kwargs...)
end

"""
    measure!(sim::Simulation,t=timeNext(sim))

Measure a dynamic `body` to update the `flow` and `pois` coefficients.
"""
function measure!(sim::AbstractSimulation,t=sum(sim.flow.Δt))
    measure!(sim.flow,sim.body;t,ϵ=sim.ϵ)
    update!(sim.pois)
end

export AbstractSimulation,Simulation,sim_step!,sim_time,measure!

# default WriteVTK functions
function vtkWriter end
function write! end
function default_attrib end
function pvd_collection end
# export
export vtkWriter, write!, default_attrib

# default ReadVTK functions
function restart_sim! end
# export
export restart_sim!

#default Plots functions
function flood end
function addbody end
function body_plot! end
function sim_gif! end
function plot_logger end
# export
export flood,addbody,body_plot!,sim_gif!,plot_logger

# Check number of threads when loading WaterLily
"""
    check_nthreads(::Val{1})

Check the number of threads available for the Julia session that loads WaterLily.
A warning is shown when running in serial (`JULIA_NUM_THREADS=1`).
"""
check_nthreads(::Val{1}) = @warn("\nUsing WaterLily in serial (ie. JULIA_NUM_THREADS=1) is not recommended because \
    it disables the GPU backend and defaults to serial CPU."*
    "\nUse JULIA_NUM_THREADS=auto, or any number of threads greater than 1, to allow multi-threading in CPU or GPU backends.")
check_nthreads(_) = nothing

# Backward compatibility for extensions
if !isdefined(Base, :get_extension)
    using Requires
end
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e" include("../ext/WaterLilyAMDGPUExt.jl")
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/WaterLilyCUDAExt.jl")
        @require WriteVTK = "64499a7a-5c06-52f2-abe2-ccb03c286192" include("../ext/WaterLilyWriteVTKExt.jl")
        @require ReadVTK = "dc215faf-f008-4882-a9f7-a79a826fadc3" include("../ext/WaterLilyReadVTKExt.jl")
        @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("../ext/WaterLilyPlotsExt.jl")
    end
    check_nthreads(Val{Threads.nthreads()}())
end

end # module

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
    Simulation(dims::NTuple, u_BC::Union{NTuple,Function}, L::Number;
               U=norm2(u_BC), Δt=0.25, ν=0., ϵ=1, perdir=()
               uλ::nothing, g=nothing, exitBC=false,
               body::AbstractBody=NoBody(),
               T=Float32, mem=Array)

Constructor for a WaterLily.jl simulation:

  - `dims`: Simulation domain dimensions.
  - `u_BC`: Simulation domain velocity boundary conditions, either a
            tuple `u_BC[i]=uᵢ, i=eachindex(dims)`, or a time and space-varying function `u_BC(i,x,t)`
  - `L`: Simulation length scale.
  - `U`: Simulation velocity scale.
  - `Δt`: Initial time step.
  - `ν`: Scaled viscosity (`Re=UL/ν`).
  - `g`: Domain acceleration, `g(i,t)=duᵢ/dt`
  - `ϵ`: BDIM kernel width.
  - `perdir`: Domain periodic boundary condition in the `(i,)` direction.
  - `exitBC`: Convective exit boundary condition in the `i=1` direction.
  - `uλ`: Function to generate the initial velocity field.
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
    function Simulation(dims::NTuple{N}, u_BC, L::Number;
                        Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, perdir=(),
                        uλ=nothing, exitBC=false, body::AbstractBody=NoBody(),
                        T=Float32, mem=Array) where N
        @assert !(isa(u_BC,Function) && isa(uλ,Function)) "`u_BC` and `uλ` cannot be both specified as Function"
        @assert !(isnothing(U) && isa(u_BC,Function)) "`U` must be specified if `u_BC` is a Function"
        hasmethod(u_BC,Tuple{Int,Number}) ? (uBC(i,x,t) = u_BC(i,t)) : uBC = u_BC # copy the tuple or the function U(i,x,t)
        isa(uBC,Function) && @assert all(typeof.(ntuple(i->uBC(i,zeros(SVector{N}),zero(T)),N)).==T) "`u_BC` is not type stable"
        uλ = isnothing(uλ) ? ifelse(isa(uBC,Function),(i,x)->uBC(i,x,zero(T)),(i,x)->uBC[i]) : uλ
        U = isnothing(U) ? √sum(abs2,uBC) : U # default if not specified
        flow = Flow(dims,uBC;uλ,Δt,ν,g,T,f=mem,perdir,exitBC)
        measure!(flow,body;ϵ)
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir))
    end
end

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
"""
function sim_step!(sim::AbstractSimulation,t_end;remeasure=true,max_steps=typemax(Int),body_force=nothing,verbose=false)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure, body_force)
        verbose && println("tU/L=",round(sim_time(sim),digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end
function sim_step!(sim::AbstractSimulation;remeasure=true,body_force=nothing)
    remeasure && measure!(sim)
    mom_step!(sim.flow, sim.pois; body_force)
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

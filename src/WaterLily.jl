"""
$(README)
"""
module WaterLily

using DocStringExtensions

include("util.jl")
export L₂,BC!,@inside,inside,δ,apply!,loc,@log,set_backend,backend,
       global_allreduce,global_dot,global_sum,global_length,global_min,global_max,
       scalar_halo!,velocity_halo!,
       comm!,velocity_comm!,pin_pressure!,
       AbstractParMode,Serial,par_mode

using Reexport
@reexport using KernelAbstractions: @kernel,@index,get_backend

# ── MPI parallel interface (implemented by WaterLilyMPIExt) ───────────────────
"""
    global_offset(Val(N), T=Float32) → SVector{N,T}

Return the global coordinate offset for this MPI rank.  Serial default returns
zero.  The MPI extension adds a method for `Parallel` that returns the rank-local
origin in global index space.
"""
global_offset(::Val{N}, ::Type{T}=Float32) where {N,T} = _global_offset(Val(N), T, par_mode[])
global_offset(N::Int, T::Type=Float32) = global_offset(Val(N), T)
_global_offset(::Val{N}, ::Type{T}, ::Serial) where {N,T} = zero(SVector{N,T})

"""
    init_waterlily_mpi(global_dims; perdir=()) → (local_dims, rank, comm)

Initialize MPI domain decomposition for WaterLily.  Implemented by `WaterLilyMPIExt`.
"""
function init_waterlily_mpi end

export global_offset, init_waterlily_mpi, mpi_rank, mpi_comm, @distributed

include("Poisson.jl")
export AbstractPoisson,Poisson,solver!,mult!

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solver!,mult!

include("Flow.jl")
export Flow,mom_step!,quick,cds

include("Body.jl")
export AbstractBody,measure_sdf!

include("AutoBody.jl")
export AutoBody,Bodies,measure,sdf,+,-

include("Metrics.jl")
export MeanFlow,update!,uu!,uu

abstract type AbstractSimulation end

include("RigidMap.jl")
export RigidMap,setmap

"""
    Simulation(dims::NTuple, uBC::Union{NTuple,Function}, L::Number;
               U=√sum(abs2,uBC), Δt=0.25, ν=0., ϵ=1, g=nothing,
               perdir=(), exitBC=false,
               body::AbstractBody=NoBody(),
               T=Float32, mem=Array)

Constructor for a WaterLily.jl simulation:

  - `dims`: Simulation domain dimensions.
  - `uBC`: Velocity field applied to boundary and acceleration conditions.
        Define a `Tuple` for constant BCs, or a `Function` for space and time varying BCs `uBC(i,x,t)`.
  - `L`: Simulation length scale.
  - `U`: Simulation velocity scale. Required if using `Uλ::Function`.
  - `Δt`: Initial time step.
  - `ν`: Scaled viscosity (`Re=UL/ν`).
  - `g`: Domain acceleration, `g(i,x,t)=duᵢ/dt`
  - `ϵ`: BDIM kernel width.
  - `perdir`: Domain periodic boundary condition in the `(i,)` direction.
  - `uλ`: Velocity field applied to the initial condition.
        Define a Tuple for homogeneous (per direction) IC, or a `Function` for space varying IC `uλ(i,x)`.
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
    function Simulation(dims::NTuple{N}, uBC, L::Number;
                        Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, perdir=(),
                        uλ=nothing, exitBC=false, body::AbstractBody=NoBody(),
                        T=Float32, mem=Array) where N
        @assert !(isnothing(U) && isa(uBC,Function)) "`U` (velocity scale) must be specified if boundary conditions `uBC` is a `Function`"
        check_mem(mem)
        isnothing(U) && (U = √sum(abs2,uBC))
        check_fn(uBC,N,T,3); check_fn(g,N,T,3); check_fn(uλ,N,T,2)
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
    sim_step!(sim::AbstractSimulation,t_end;remeasure=true,λ=quick,max_steps=typemax(Int),verbose=false,
        udf=nothing,kwargs...)

Integrate the simulation `sim` up to dimensionless time `t_end`.
If `remeasure=true`, the body is remeasured at every time step. Can be set to `false` for static geometries to speed up simulation.
A user-defined function `udf` can be passed to arbitrarily modify the `::Flow` during the predictor and corrector steps.
If the `udf` user keyword arguments, these needs to be included in the `sim_step!` call as well.
A `λ::Function` function can be passed as a custom convective scheme, following the interface of `λ(u,c,d)` (for upstream, central,
downstream points).
"""
function sim_step!(sim::AbstractSimulation,t_end;remeasure=true,λ=quick,max_steps=typemax(Int),verbose=false,
        udf=nothing,kwargs...)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure, λ, udf, kwargs...)
        verbose && sim_info(sim)
    end
end
function sim_step!(sim::AbstractSimulation;remeasure=true,λ=quick,udf=nothing,kwargs...)
    remeasure && measure!(sim)
    mom_step!(sim.flow, sim.pois; λ, udf, kwargs...)
end

"""
    measure!(sim::Simulation,t=timeNext(sim))

Measure a dynamic `body` to update the `flow` and `pois` coefficients.
"""
function measure!(sim::AbstractSimulation,t=sum(sim.flow.Δt))
    measure!(sim.flow,sim.body;t,ϵ=sim.ϵ)
    sim.pois.levels[1].L .= sim.flow.μ₀
    update!(sim.pois)
end

"""
    sim_info(sim::AbstractSimulation)
Prints information on the current state of a simulation.
"""
sim_info(sim::AbstractSimulation) = println("tU/L=",round(sim_time(sim),digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))

"""
    perturb!(sim; noise=0.1)
Perturb the velocity field of a simulation with `noise` level with respect to velocity scale `U`.
"""
perturb!(sim::AbstractSimulation; noise=0.1) = sim.flow.u .+= randn(size(sim.flow.u))*sim.U*noise |> typeof(sim.flow.u).name.wrapper

export AbstractSimulation,Simulation,sim_step!,sim_time,measure!,sim_info,perturb!

# defaults JLD2 and VTK I/O functions
function load!(sim::AbstractSimulation; kwargs...)
    fname = get(Dict(kwargs), :fname, "WaterLily.jld2")
    ext = split(fname, ".")[end] |> Symbol
    vtk_loaded = !isnothing(Base.get_extension(WaterLily, :WaterLilyReadVTKExt))
    jld2_loaded = !isnothing(Base.get_extension(WaterLily, :WaterLilyJLD2Ext))
    ext == :pvd && (@assert vtk_loaded "WriteVTK must be loaded to save .pvd data.")
    ext == :jdl2 && (@assert jld2_loaded "JLD2 must be loaded to save .jld2 data.")
    load!(sim, Val{ext}(); kwargs...)
end
function save! end
function vtkWriter end
function default_attrib end
function pvd_collection end
export load!, save!, vtkWriter, default_attrib

# default Plots functions
function flood end
function addbody end
function body_plot! end
function sim_gif! end
function plot_logger end
# export
export flood,addbody,body_plot!,sim_gif!,plot_logger

# default Makie functions
function viz! end
function get_body end
function plot_body_obs! end
# export
export viz!, get_body, plot_body_obs!

# Check number of threads when loading WaterLily
"""
    check_nthreads()

Check the number of threads available for the Julia session that loads WaterLily.
A warning is shown when running in serial (JULIA_NUM_THREADS=1) with KernelAbstractions enabled.
Skipped during precompilation (which always runs single-threaded).
"""
function check_nthreads()
    ccall(:jl_generating_output, Cint, ()) != 0 && return  # skip during precompilation
    if backend == "KernelAbstractions" && Threads.nthreads() == 1
        @warn """
        Using WaterLily in serial (ie. JULIA_NUM_THREADS=1) is not recommended because it defaults to serial CPU execution.
        Use JULIA_NUM_THREADS=auto, or any number of threads greater than 1, to allow multi-threading in CPU backends.
        For a low-overhead single-threaded CPU only backend set: WaterLily.set_backend("SIMD")"""
    end
end
function __init__()
    check_nthreads()
end

"""
    check_mem(mem)

Check that the memory type `mem` is compatible with the current backend.
GPU array types (anything other than `Array`) require the KernelAbstractions backend.
"""
function check_mem(mem)
    if backend == "SIMD" && mem !== Array
        error("GPU memory (mem=$mem) requires the KernelAbstractions backend. " *
              "Set it with: WaterLily.set_backend(\"KernelAbstractions\") then restart the Julia session.")
    end
end

end # module

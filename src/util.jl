using KernelAbstractions: get_backend, @index, @kernel
using LoggingExtras
using LinearAlgebra: ⋅

# custom log macro
_psolver = Logging.LogLevel(-123) # custom log level for pressure solver, needs the negative sign
macro log(exs...)
    quote
        @logmsg _psolver $(map(x -> esc(x), exs)...)
    end
end
"""
    logger(fname="WaterLily")

Set up a logger to write the pressure solver data to a logging file named `WaterLily.log`.
"""
function logger(fname::String="WaterLily")
    ENV["JULIA_DEBUG"] = all
    logger = FormatLogger(ifelse(fname[end-3:end]==".log",fname[1:end-4],fname)*".log"; append=false) do io, args
        args.level == _psolver && print(io, args.message)
    end;
    global_logger(logger);
    # put header in file
    @log "p/c, iter, r∞, r₂, ω\n"
end

@inline CI(a...) = CartesianIndex(a...)
"""
    CIj(j,I,k)
Replace jᵗʰ component of CartesianIndex with k
"""
CIj(j,I::CartesianIndex{d},k) where d = CI(ntuple(i -> i==j ? k : I[i], d))

"""
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

"""
    inside(a;buff=1)

Return CartesianIndices range excluding `buff` layers of cells on all boundaries.
Default `buff=1` matches the N+2 staggered grid layout (1 ghost cell per side).
"""
@inline inside(a::AbstractArray;buff=1) = CartesianIndices(map(ax->first(ax)+buff:last(ax)-buff,axes(a)))

"""
    inside_u(dims,j)

Return CartesianIndices range excluding the ghost-cells on the boundaries of
a _vector_ array on face `j` with size `dims`.
"""
function inside_u(dims::NTuple{N},j) where {N}
    CartesianIndices(ntuple( i-> i==j ? (3:dims[i]-1) : (2:dims[i]), N))
end
@inline inside_u(dims::NTuple{N}) where N = CartesianIndices((map(i->(2:i-1),dims)...,1:N))
@inline inside_u(u::AbstractArray) = CartesianIndices(map(i->(2:i-1),size(u)[1:end-1]))
splitn(n) = Base.front(n),last(n)
size_u(u) = splitn(size(u))

local_dot(a, b) = a⋅b
local_sum(a) = sum(a)

"""
    @inside <expr>

Simple macro to automate efficient loops over cells excluding ghosts. For example,

    @inside p[I] = sum(loc(0,I))

becomes

    @loop p[I] = sum(loc(0,I)) over I ∈ inside(p)

See [`@loop`](@ref).
"""
macro inside(ex)
    # Make sure it's a single assignment
    @assert ex.head == :(=) && ex.args[1].head == :(ref)
    a,I = ex.args[1].args[1:2]
    return quote # loop over the size of the reference
        WaterLily.@loop $ex over $I ∈ inside($a)
    end |> esc
end

# Could also use ScopedValues in Julia 1.11+
using Preferences
const backend = @load_preference("backend", "KernelAbstractions")
"""
    set_backend(new_backend::String)

Set the loop execution backend to `"SIMD"` (single-threaded) or
`"KernelAbstractions"` (multi-threaded CPU / GPU).  The preference is
persisted via Preferences.jl; a Julia restart is required.
"""
function set_backend(new_backend::String)
    if !(new_backend in ("SIMD", "KernelAbstractions"))
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend" => new_backend)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

"""
    @loop <expr> over <I ∈ R>

Macro to automate fast loops using @simd when running in serial,
or KernelAbstractions when running multi-threaded CPU or GPU.

For example

    @loop a[I,i] += sum(loc(i,I)) over I ∈ R

becomes

    @simd for I ∈ R
        @fastmath @inbounds a[I,i] += sum(loc(i,I,offset))
    end

on serial execution, or

    @kernel function kern(a,i,@Const(offset),@Const(I0))
        I ∈ @index(Global,Cartesian)+I0
        @fastmath @inbounds a[I,i] += sum(loc(i,I,offset))
    end
    kern(get_backend(a),64)(a,i,offset,R[1]-oneunit(R[1]),ndrange=size(R))

when multi-threading on CPU or using CuArrays.  The macro rewrites every
`loc(...)` call in `expr` to append a captured `offset` argument so that
`loc` returns *global* coordinates in MPI-parallel runs; in serial the
captured value is `nothing` and `loc(...,nothing)` falls back to local
coordinates.  `get_backend` is used on the _first_ variable in `expr`.
"""
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args
    sym = []
    grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    symT = [gensym() for _ in 1:length(sym)] # generate a list of types for each symbol
    symWtypes = joinsymtype(rep.(sym),symT) # symbols with types: [a::A, b::B, ...]
    @gensym(kern, kern_, offset) # unique kernel names + captured offset symbol
    inject_loc_offset!(ex, offset) # rewrite loc(...) → loc(..., offset) in ex
    @static if backend == "KernelAbstractions"
        return quote
            local $offset = WaterLily._loop_offset(eltype($(sym[1])))
            @kernel function $kern_($(symWtypes...),@Const($offset),@Const(I0)) where {$(symT...)}
                $I = @index(Global,Cartesian)
                $I += I0
                @fastmath @inbounds $ex
            end
            function $kern($(symWtypes...),$offset) where {$(symT...)}
                $kern_(get_backend($(sym[1])),64)($(sym...),$offset,$R[1]-oneunit($R[1]),ndrange=size($R))
            end
            $kern($(sym...),$offset)
        end |> esc
    else # backend == "SIMD"
        return quote
            local $offset = WaterLily._loop_offset(eltype($(sym[1])))
            function $kern($(symWtypes...),$offset) where {$(symT...)}
                @simd for $I ∈ $R
                    @fastmath @inbounds $ex
                end
            end
            $kern($(sym...),$offset)
        end |> esc
    end
end
function grab!(sym,ex::Expr)
    ex.head == :. && return union!(sym,[ex])      # grab composite name and return
    start = ex.head==:(call) ? 2 : 1              # don't grab function names
    foreach(a->grab!(sym,a),ex.args[start:end])   # recurse into args
    ex.args[start:end] = rep.(ex.args[start:end]) # replace composites in args
end
grab!(sym,ex::Symbol) = union!(sym,[ex])          # grab symbol name
grab!(sym,ex) = nothing
rep(ex) = ex
rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[2].value) : ex
joinsymtype(sym::Symbol,symT::Symbol) = Expr(:(::), sym, symT)
joinsymtype(sym,symT) = zip(sym,symT) .|> x->joinsymtype(x...)

# Walk `ex` and append `offset` to every bare `loc(...)` call so positions
# inside a @loop body are in global coordinates.  The rewrite uses
# `GlobalRef(@__MODULE__, :loc)` so the call resolves unambiguously to
# WaterLily.loc, even if the caller's module does not `using WaterLily:loc`.
# Scope-aware: `function`, lambda (`->`), `let`, and `for` scopes that bind
# the symbol `loc` disable the rewrite within their body.  Qualified names
# like `WaterLily.loc` are untouched because their head is `:.`, not `:call`.
function inject_loc_offset!(ex, offset, shadowed::Bool=false)
    ex isa Expr || return ex
    h = ex.head
    if h === :function || h === :->
        inner = shadowed || _binds_loc(ex.args[1])
        inject_loc_offset!(ex.args[end], offset, inner)
        return ex
    elseif h === :(=) && ex.args[1] isa Expr && ex.args[1].head === :call
        # short-form fn def: f(x) = body
        inner = shadowed || _binds_loc(ex.args[1])
        inject_loc_offset!(ex.args[2], offset, inner)
        return ex
    elseif h === :let || h === :for
        bind, body = ex.args[1], ex.args[end]
        inject_loc_offset!(bind, offset, shadowed)
        inner = shadowed || _binds_loc(bind)
        inject_loc_offset!(body, offset, inner)
        return ex
    end
    if h === :call && ex.args[1] === :loc && !shadowed
        ex.args[1] = GlobalRef(@__MODULE__, :loc)
        push!(ex.args, offset)
    end
    foreach(a -> inject_loc_offset!(a, offset, shadowed), ex.args)
    return ex
end

# Does this LHS expression bind the symbol `loc`?  Used to detect scopes
# (function/lambda params, let/for bindings) that shadow WaterLily.loc.
_binds_loc(s::Symbol) = s === :loc
function _binds_loc(ex::Expr)
    h = ex.head
    if h === :tuple || h === :block || h === :call
        any(_binds_loc, ex.args)
    elseif h === :(::) || h === :(=) || h === :kw || h === :...
        _binds_loc(ex.args[1])
    else
        false
    end
end
_binds_loc(_) = false

using StaticArrays
"""
    loc(i,I) = loc(Ii)

Location in space of the cell at CartesianIndex `I` at face `i`.
Using `i=0` returns the cell center s.t. `loc(0,I) = I .- 1.5`.

Inside a `@loop` body the macro automatically appends the MPI rank-local
offset so `loc(...)` returns *global* coordinates — user code is identical
in serial and parallel.  Outside `@loop`, `loc(...)` returns rank-local
coordinates; add `global_offset(Val(N), T)` explicitly to get global ones.
"""
@inline loc(i,I::CartesianIndex{N},T::Type=Float32) where N = SVector{N,T}(I.I .- 1.5 .- 0.5 .* δ(i,I).I)
@inline loc(Ii::CartesianIndex,T::Type=Float32) = loc(last(Ii),Base.front(Ii),T)
# SVector offset overloads — used by the @loop auto-injection
@inline loc(i,I::CartesianIndex{N},offset::SVector{N,T}) where {N,T} = loc(i,I,T) + offset
@inline loc(Ii::CartesianIndex,offset::SVector) = loc(last(Ii),Base.front(Ii),eltype(offset)) + offset
@inline loc(i,I::CartesianIndex{N},T::Type,offset::SVector{N}) where N = loc(i,I,T) + offset
@inline loc(Ii::CartesianIndex,T::Type,offset::SVector) = loc(last(Ii),Base.front(Ii),T) + offset
# Nothing sentinel: serial @loop passes `nothing` — no-ops to the plain loc
@inline loc(i,I::CartesianIndex,::Nothing) = loc(i,I)
@inline loc(Ii::CartesianIndex,::Nothing) = loc(Ii)
@inline loc(i,I::CartesianIndex,T::Type,::Nothing) = loc(i,I,T)
@inline loc(Ii::CartesianIndex,T::Type,::Nothing) = loc(Ii,T)
Base.last(I::CartesianIndex) = last(I.I)
Base.front(I::CartesianIndex) = CI(Base.front(I.I))
"""
    apply!(f, c)

Apply a vector function `f(i,x)` to the faces of a uniform staggered array `c` or
a function `f(x)` to the center of a uniform array `c`.
"""
apply!(f,c) = hasmethod(f,Tuple{Int,CartesianIndex}) ? applyV!(f,c) : applyS!(f,c)
applyV!(f,c) = @loop c[Ii] = f(last(Ii),loc(Ii,eltype(c))) over Ii ∈ CartesianIndices(c)
applyS!(f,c) = @loop c[I] = f(loc(0,I,eltype(c))) over I ∈ CartesianIndices(c)
"""
    slice(dims,i,j,low=1)

Return `CartesianIndices` range slicing through an array of size `dims` in
dimension `j` at index `i` (or range `i`). `low` optionally sets the lower
extent of the range in the other dimensions.
"""
function slice(dims::NTuple{N},i,j,low=1) where N
    CartesianIndices(ntuple( k-> k==j ? (i:i) : (low:dims[k]), N))
end
function slice(dims::NTuple{N},i::AbstractUnitRange,j,low=1) where N
    CartesianIndices(ntuple( k-> k==j ? i : (low:dims[k]), N))
end

# ── AbstractParMode dispatch pattern ──────────────────────────────────────────
#
# Serial WaterLily dispatches all hooks through `par_mode[]` (defaults to Serial()).
# The MPI extension adds `Parallel <: AbstractParMode` with MPI-aware methods —
# no method overwriting, so precompilation works normally.
abstract type AbstractParMode end
struct Serial <: AbstractParMode end
const par_mode = Ref{AbstractParMode}(Serial())

"""
    mpi_rank() → Int
    mpi_comm() → Union{Nothing,MPI.Comm}

Rank accessors available in any mode: serial returns `0` / `nothing`; under MPI
the extension returns the rank and communicator stored on `par_mode[]::Parallel`.
Useful for rank-gated printing (`mpi_rank() == 0 && @info ...`) without having
to thread `(me, comm)` through user scope.
"""
mpi_rank() = _mpi_rank(par_mode[])
mpi_comm() = _mpi_comm(par_mode[])
_mpi_rank(::Serial) = 0
_mpi_comm(::Serial) = nothing

"""
    @distributed Simulation(dims, uBC, L; kwargs...)
    @distributed sim = Simulation(dims, uBC, L; kwargs...)

Boilerplate-free MPI initialization for a WaterLily `Simulation`.  The macro
pulls the global `dims` and (optional) `perdir` kwarg from the `Simulation`
call, invokes `init_waterlily_mpi(dims; perdir=perdir)`, and substitutes the
returned rank-local dimensions back into the `Simulation` call.  Requires
`using MPI, ImplicitGlobalGrid` so the extension is loaded; otherwise the
generated `init_waterlily_mpi` call throws `MethodError`.  The user is still
responsible for `finalize_global_grid()` at script end.

Example:

    using WaterLily, MPI, ImplicitGlobalGrid
    sim = @distributed Simulation((192, 128), (U, 0), L;
                                  ν=ν, body=body, perdir=(1,2))
    mpi_rank() == 0 && @info "decomposed and ready"
    # ... time stepping ...
    finalize_global_grid()
"""
macro distributed(ex)
    if ex isa Expr && ex.head === :(=)
        lhs, rhs = ex.args[1], ex.args[2]
        return esc(Expr(:(=), lhs, _rewrite_distributed_call(rhs)))
    else
        return esc(_rewrite_distributed_call(ex))
    end
end

function _rewrite_distributed_call(ex)
    (ex isa Expr && ex.head === :call && ex.args[1] === :Simulation) ||
        error("@distributed expects a `Simulation(...)` call, got $(ex)")

    has_params = length(ex.args) >= 2 && ex.args[2] isa Expr && ex.args[2].head === :parameters
    dims_idx   = has_params ? 3 : 2
    length(ex.args) >= dims_idx ||
        error("@distributed: `Simulation` call missing positional `dims` argument")
    global_dims = ex.args[dims_idx]

    perdir = :(())
    if has_params
        for kw in ex.args[2].args
            if kw isa Expr && kw.head === :kw && kw.args[1] === :perdir
                perdir = kw.args[2]
                break
            end
        end
    end

    local_sym = gensym(:local_dims)
    new_args = copy(ex.args)
    new_args[dims_idx] = local_sym
    sim_call = Expr(:call, new_args...)

    quote
        $(local_sym), _, _ = init_waterlily_mpi($(global_dims); perdir=$(perdir))
        $(sim_call)
    end
end

"""
    _loop_offset(::Type{T})

Return the offset captured into `@loop` bodies so `loc(...)` calls inside the
expression return global coordinates.  Serial returns `nothing` (no-op — the
`loc(..., ::Nothing)` overload falls back to plain `loc(...)`); the MPI
extension returns an `SVector{N,T}` rank-local offset.
"""
_loop_offset(::Type{T}) where T = _loop_offset(T, par_mode[])
_loop_offset(::Type{T}, ::Serial) where T = nothing

"""
    global_dot(a, b)

Global dot product `a⋅b`.  In serial, equivalent to `a⋅b`.
The MPI extension replaces this with `MPI.Allreduce(…, SUM)`.
"""
global_dot(a, b) = global_allreduce(local_dot(a, b))
"""
    global_sum(a)

Global sum of array `a`.  MPI-aware via dispatch on `par_mode[]`.
"""
global_sum(a) = global_allreduce(local_sum(a))
"""
    global_length(r)

Global length of index range `r`.  MPI-aware via dispatch on `par_mode[]`.
"""
global_length(r) = global_allreduce(length(r))
"""
    global_min(a, b)

Global minimum of `a` and `b`.  MPI-aware via dispatch on `par_mode[]`.
"""
global_min(a, b) = _global_min(a, b, par_mode[])

_global_min(a, b, ::Serial) = min(a, b)

"""
    global_max(x)

Global maximum of scalar `x` across ranks.  MPI-aware via dispatch on `par_mode[]`.
"""
global_max(x) = _global_max(x, par_mode[])

_global_max(x, ::Serial) = x

"""
    global_allreduce(x)

Reduce a pre-computed value `x` (scalar or vector) across all MPI ranks
with element-wise summation.  In serial, returns `x` unchanged.
This is the primitive that other global reductions build on:
`global_sum(a) = global_allreduce(local_sum(a))`.
"""
global_allreduce(x) = _global_allreduce(x, par_mode[])
_global_allreduce(x, ::Serial) = x

"""
    L₂(a)

L₂ norm of array `a` over interior cells (excluding ghost cells).
"""
L₂(a) = (R = inside(a); @view(a[R])⋅@view(a[R]))

"""
    global_perdot(a, b, perdir)

Dot product of `a` and `b` respecting periodic boundary conditions.
When `perdir` is empty, uses the full arrays; otherwise restricts to interior cells.
MPI-aware via dispatch on `par_mode[]`.
"""
local_perdot(a,b,::Tuple{}) = a⋅b
local_perdot(a,b,perdir,R=inside(a)) = @view(a[R])⋅@view(b[R])
global_perdot(a,b,tup::Tuple{}) = global_allreduce(local_perdot(a, b, tup))
global_perdot(a,b,perdir,R=inside(a)) = global_allreduce(local_perdot(a, b, perdir, R))

"""
    scalar_halo!(x)

Exchange halo cells for scalar array `x`.  No-op in serial.
The MPI extension routes fine-grid arrays through IGG `update_halo!`
and coarse multigrid arrays through direct `MPI.Isend`/`MPI.Irecv!`.
"""
scalar_halo!(x) = _scalar_halo!(x, par_mode[])
"""
    velocity_halo!(u)

Exchange halo cells for a velocity (vector) array `u`.  No-op in serial.
The MPI extension exchanges each component separately via `scalar_halo!`.
"""
velocity_halo!(u) = _velocity_halo!(u, par_mode[])
_scalar_halo!(x, ::Serial) = nothing
_velocity_halo!(u, ::Serial) = nothing


"""
    divisible(N)

Check if array dimension `N` is divisible for multigrid coarsening.
"""
divisible(N) = _divisible(N, par_mode[])
_divisible(N, ::Serial) = mod(N,2)==0 && N>4

"""
    BC!(a, uBC, saveexit=false, perdir=(), t=0)

Apply domain boundary conditions to the ghost cells of a _vector_ field.
A Dirichlet condition is applied to the _normal_ component; zero Neumann to tangential.
Periodic directions are handled by `velocity_comm!` (called at the end),
separating domain BCs from communication BCs.
"""
BC!(a,U,saveexit=false,perdir=(),t=0) = BC!(a,(i,x,t)->U[i],saveexit,perdir,t)
BC!(a,uBC::Function,saveexit=false,perdir=(),t=0) = _BC!(a, uBC, saveexit, perdir, t, par_mode[])
function _BC!(a, uBC::Function, saveexit, perdir, t, ::Serial)
    N,n = size_u(a)
    for i ∈ 1:n, j ∈ 1:n
        j in perdir && continue  # periodic handled by velocity_comm!
        if i==j # Normal direction, Dirichlet
            for s ∈ (1,2)
                @loop a[I,i] = uBC(i,loc(i,I),t) over I ∈ slice(N,s,j)
            end
            (!saveexit || i>1) && (@loop a[I,i] = uBC(i,loc(i,I),t) over I ∈ slice(N,N[j],j))
        else    # Tangential directions, Neumann: mirror
            @loop a[I,i] = uBC(i,loc(i,I),t)+a[I+δ(j,I),i]-uBC(i,loc(i,I+δ(j,I)),t) over I ∈ slice(N,1,j)
            @loop a[I,i] = uBC(i,loc(i,I),t)+a[I-δ(j,I),i]-uBC(i,loc(i,I-δ(j,I)),t) over I ∈ slice(N,N[j],j)
        end
    end
    velocity_comm!(a, perdir)
end

"""
    exitBC!(u,u⁰,Δt)

Apply a 1D convection scheme to fill the ghost cell on the exit of the domain.
"""
exitBC!(u,u⁰,Δt) = _exitBC!(u,u⁰,Δt,par_mode[])
function _exitBC!(u,u⁰,Δt,::Serial)
    N,_ = size_u(u)
    exitR = slice(N.-1,N[1],1,2)              # exit slice excluding ghosts (right wall face)
    U = sum(@view(u[slice(N.-1,2,1,2),1]))/length(exitR) # inflow mass flux (left wall face)
    @loop u[I,1] = u⁰[I,1]-U*Δt*(u⁰[I,1]-u⁰[I-δ(1,I),1]) over I ∈ exitR
    ∮u = sum(@view(u[exitR,1]))/length(exitR)-U   # mass flux imbalance
    @loop u[I,1] -= ∮u over I ∈ exitR         # correct flux
end
"""
    perBC!(a,perdir)

Apply periodic conditions to the ghost cells of a _scalar_ field.
"""
perBC!(a,::Tuple{}) = nothing
perBC!(a, perdir, N = size(a)) = for j ∈ perdir
    @loop a[I] = a[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
    @loop a[I] = a[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
end

"""
    comm!(a, perdir)

Scalar communication: periodic BC copy + MPI halo exchange.
In serial, just applies `perBC!`. In parallel, MPI halo handles everything.
"""
comm!(a, perdir) = _comm!(a, perdir, par_mode[])
_comm!(a, perdir, ::Serial) = perBC!(a, perdir)

"""
    velocity_comm!(a, perdir)

Velocity communication: periodic BC copy + MPI halo exchange.
In serial, copies periodic ghost cells for all velocity components.
In parallel, MPI halo handles everything.
"""
velocity_comm!(a, perdir) = _velocity_comm!(a, perdir, par_mode[])
function _velocity_comm!(a, perdir, ::Serial)
    _, n = size_u(a)
    for i ∈ 1:n
        perBC!(@view(a[..,i]), perdir)
    end
end

"""
    interp(x::SVector, arr::AbstractArray, offset=zero(x))

Linear interpolation from array `arr` at Cartesian-coordinate `x`.
`offset` shifts `x` before indexing — used in MPI parallel to map
global coordinates to rank-local array indices (pass `flow.offset`).
"""
function interp(x::SVector{D,T}, arr::AbstractArray{T,D}, offset=zero(x)) where {D,T}
    # Index below the interpolation coordinate and the difference
    x = x .- offset .+ 1.5f0; i = floor.(Int,x); y = x.-i

    # CartesianIndices around x
    I = CartesianIndex(i...); R = I:I+oneunit(I)

    # Linearly weighted sum over arr[R] (in serial)
    s = zero(T)
    @fastmath @inbounds @simd for J in R
        weight = prod(@. ifelse(J.I==I.I,1-y,y))
        s += arr[J]*weight
    end
    return s
end
using EllipsisNotation
function interp(x::SVector{D,T}, varr::AbstractArray{T}, offset=zero(x)) where {D,T}
    # Shift to align with each staggered grid component and interpolate
    @inline shift(i) = SVector{D,T}(ifelse(i==j,0.5,0.) for j in 1:D)
    return SVector{D,T}(interp(x+shift(i),@view(varr[..,i]),offset) for i in 1:D)
end

"""
    sgs!(flow, t; νₜ, S, Cs, Δ)

Implements a user-defined function `udf` to model subgrid-scale LES stresses based on the Boussinesq approximation
    τᵃᵢⱼ = τʳᵢⱼ - (1/3)τʳₖₖδᵢⱼ = -2νₜS̅ᵢⱼ
where
            ▁▁▁▁
    τʳᵢⱼ =  uᵢuⱼ - u̅ᵢu̅ⱼ

and we add -∂ⱼ(τᵃᵢⱼ) to the RHS as a body force (the isotropic part of the tensor is automatically modelled by the pressure gradient term).
Users need to define the turbulent viscosity function `νₜ` and pass it as a keyword argument to this function together with rate-of-strain
tensor array buffer `S`, Smagorinsky constant `Cs`, and filter width `Δ`.
For example, the standard Smagorinsky–Lilly model for the sub-grid scale stresses is

    νₜ = (CₛΔ)²|S̅ᵢⱼ|=(CₛΔ)²√(2S̅ᵢⱼS̅ᵢⱼ)

It can be implemented as
    `smagorinsky(I::CartesianIndex{m} where m; S, Cs, Δ) = @views (Cs*Δ)^2*sqrt(dot(S[I,:,:],S[I,:,:]))`
and passed into `sim_step!` as a keyword argument together with the variables that the function needs (`S`, `Cs`, and `Δ`):
    `sim_step!(sim, ...; udf=sgs, νₜ=smagorinsky, S, Cs, Δ)`
"""
function sgs!(flow, t; νₜ, S, Cs, Δ)
    N,n = size_u(flow.u)
    @loop S[I,:,:] .= WaterLily.S(I,flow.u) over I ∈ inside(flow.σ)
    for i ∈ 1:n, j ∈ 1:n
        WaterLily.@loop (
            flow.σ[I] = -νₜ(I;S,Cs,Δ)*∂(j,CI(I,i),flow.u);
            flow.f[I,i] += flow.σ[I];
        ) over I ∈ inside_u(N,j)
        WaterLily.@loop flow.f[I-δ(j,I),i] -= flow.σ[I] over I ∈ WaterLily.inside_u(N,j)
    end
end

check_fn(f,N,T,nargs) = nothing
function check_fn(f::Function,N,T,nargs)
    @assert first(methods(f)).nargs==nargs+1 "$f signature needs $nargs arguments"
    @assert all(typeof.(ntuple(i->f(i,xtargs(Val{}(nargs),N,T)...),N)).==T) "$f is not type stable"
end
xtargs(::Val{2},N,T) = (zeros(SVector{N,T}),)
xtargs(::Val{3},N,T) = (zeros(SVector{N,T}),zero(T))

ic_function(uBC::Function) = (i,x)->uBC(i,x,0)
ic_function(uBC::Tuple) = (i,x)->uBC[i]

squeeze(a::AbstractArray) = dropdims(a, dims = tuple(findall(size(a) .== 1)...))


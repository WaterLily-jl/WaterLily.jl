# ── AbstractParMode dispatch pattern ──────────────────────────────────────────
#
# Serial WaterLily dispatches all parallel hooks through `par_mode[]` (defaults
# to Serial()). The MPI extension (WaterLilyMPIExt) adds `Parallel <:
# AbstractParMode` with MPI-aware methods — no method overwriting, so
# precompilation works normally. This file only depends on Base/stdlib and is
# included before core.jl (whose `_BC!`/`_exitBC!` signatures reference
# `AbstractParMode`).
using LinearAlgebra: ⋅
using StaticArrays
using EllipsisNotation

abstract type AbstractParMode end
struct Serial <: AbstractParMode end
const par_mode = Ref{AbstractParMode}(Serial())

# Rank-local reduction kernels the global reductions build on.
local_dot(a, b) = a⋅b
local_sum(a) = sum(a)
local_sumabs(a) = sum(abs, a)  # Σ|aᵢ| — L₁ bulk term of the Poisson stopping criterion

"""
    mpi_rank()   → Int
    mpi_comm()   → Union{Nothing,MPI.Comm}
    mpi_nprocs() → Int

Rank accessors available in any mode: serial returns `0` / `nothing` / `1`;
under MPI the extension returns the rank, communicator, and rank count stored
on `par_mode[]::Parallel`. Useful for rank-gated printing
(`mpi_rank() == 0 && @info ...`) and rank-count-aware iteration bounds without
threading `(me, comm, np)` through user scope.
"""
mpi_rank()   = _mpi_rank(par_mode[])
mpi_comm()   = _mpi_comm(par_mode[])
mpi_nprocs() = _mpi_nprocs(par_mode[])
_mpi_rank(::Serial)   = 0
_mpi_comm(::Serial)   = nothing
_mpi_nprocs(::Serial) = 1

"""
    @distributed Simulation(dims, uBC, L; kwargs...)
    @distributed sim = Simulation(dims, uBC, L; kwargs...)

Boilerplate-free MPI initialization for a WaterLily simulation.  The macro
pulls the global `dims` (first positional argument) and the optional `perdir`
keyword out of the constructor call, invokes `init_waterlily_mpi(dims; perdir)`,
and substitutes the returned rank-local dimensions back into the call.
Requires `using MPI, ImplicitGlobalGrid` so the extension is loaded; otherwise
the generated `init_waterlily_mpi` call throws `MethodError`.  The user is
still responsible for `finalize_global_grid()` at script end.

Any constructor that follows `Simulation`'s calling convention is accepted, not
only `Simulation` itself: the first positional argument must be the global
`dims` tuple, and periodicity must be visible as a `perdir` keyword in the call
(`f(dims, ...; perdir=(1,))`, the shorthand `f(dims, ...; perdir)`, or
`f(dims, ..., perdir=(1,))`; a splatted `kwargs...` is not inspected).
Downstream `AbstractSimulation` constructors such as
`BiotSimulation(dims, uBC, L; ...)` therefore work unchanged.

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

_iskw(ex) = ex isa Expr && ex.head === :kw

function _rewrite_distributed_call(ex)
    (ex isa Expr && ex.head === :call) ||
        error("@distributed expects a constructor call `f(dims, ...; perdir=...)` " *
              "such as `Simulation(dims, uBC, L; kwargs...)`, got $(ex)")
    f = ex.args[1]
    # `f(dims, ...; kw...)` parses the `;` keywords into a leading :parameters
    # block; `f(dims, ..., kw=...)` leaves them as trailing :kw arguments.
    has_params = length(ex.args) >= 2 && ex.args[2] isa Expr && ex.args[2].head === :parameters
    dims_idx   = has_params ? 3 : 2
    (length(ex.args) >= dims_idx && !_iskw(ex.args[dims_idx])) ||
        error("@distributed: `$(f)(...)` has no positional `dims` argument (it must come first)")
    global_dims = ex.args[dims_idx]

    # `perdir` may be `; perdir=(1,)`, the shorthand `; perdir`, or `, perdir=(1,)`
    perdir = :(())
    for kw in (has_params ? ex.args[2].args : ())
        kw === :perdir && (perdir = :perdir; break)
        _iskw(kw) && kw.args[1] === :perdir && (perdir = kw.args[2]; break)
    end
    for kw in ex.args[dims_idx+1:end]
        _iskw(kw) && kw.args[1] === :perdir && (perdir = kw.args[2]; break)
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
    scalar_halo!(x)

Exchange halo cells for scalar array `x`.  No-op in serial.
The MPI extension routes fine-grid arrays through IGG `update_halo!`
and coarse multigrid arrays through per-size IGG grids or `MPI.Sendrecv!`.
"""
scalar_halo!(x) = _scalar_halo!(x, par_mode[])
"""
    velocity_halo!(u)

Exchange halo cells for a velocity (vector) array `u`.  No-op in serial.
The MPI extension exchanges all components in one batched call.
"""
velocity_halo!(u) = _velocity_halo!(u, par_mode[])
_scalar_halo!(x, ::Serial) = nothing
_velocity_halo!(u, ::Serial) = nothing

"""
    phys_left(j), phys_right(j)

Whether direction `j`'s left/right ghost lies on a physical domain wall.
Always `true` in serial; in MPI, `false` at rank-internal boundaries (where
halo exchange supplies neighbor data instead).
"""
phys_left(j)  = _phys_left(j, par_mode[])
phys_right(j) = _phys_right(j, par_mode[])
_phys_left(j, ::Serial)  = true
_phys_right(j, ::Serial) = true

"""
    comm!(a, perdir)

Scalar communication: periodic BC copy + MPI halo exchange.
In serial, just applies `perBC!`. In parallel, `perBC!` handles periodic
dims owned in full by the rank and the MPI halo handles decomposed dims.
"""
comm!(a, perdir) = _comm!(a, perdir, par_mode[])
_comm!(a, perdir, ::Serial) = perBC!(a, perdir)

"""
    velocity_comm!(a, perdir)

Velocity communication: periodic BC copy + MPI halo exchange.
In serial, copies periodic ghost cells for all velocity components.
"""
velocity_comm!(a, perdir) = _velocity_comm!(a, perdir, par_mode[])
function _velocity_comm!(a, perdir, ::Serial)
    _, n = size_u(a)
    for i ∈ 1:n
        perBC!(@view(a[..,i]), perdir)
    end
end

"""
    decomposed(j)

Whether direction `j` is split across MPI ranks (`true`) or owned in full
by every rank (`false`).  Always `false` in serial.
"""
decomposed(j) = _decomposed(j, par_mode[])
_decomposed(j, ::Serial) = false

"""
    effective_perdir(perdir)

Return directions where periodic wrap can be handled via explicit `N[j]-2`
indexing on the local array.  In serial this is the full `perdir`.  In MPI,
directions decomposed across ranks are excluded — the halo (halowidth=1)
cannot supply the 2-cells-back stencil required by `ϕuP`, so `conv_diff!`
falls back to the non-periodic boundary scheme (`ϕuL`/`ϕuR`) that only
needs 1 ghost cell (filled by halo from the periodic neighbor).
"""
effective_perdir(perdir) = Tuple(j for j in perdir if !decomposed(j))

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

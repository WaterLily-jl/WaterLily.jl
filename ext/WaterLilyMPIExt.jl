"""
WaterLilyMPIExt — Julia package extension
==========================================
Activated automatically when ImplicitGlobalGrid and MPI are loaded alongside
WaterLily.  Provides MPI-aware overrides for global reductions, halo exchange,
and boundary conditions at MPI-subdomain interfaces.

The serial WaterLily code uses hook functions (global_dot, global_sum,
global_length, global_min, scalar_halo!, velocity_halo!) that are no-ops
in serial.  This extension overrides them with MPI.Allreduce and halo
exchange, eliminating the need for MPI-specific overrides of pcg!,
residual!, L₂, increment!, solver!, Vcycle!, CFL, and pin_pressure!.

Functions still overridden here (beyond hooks):
  BC!         — serial BC! + velocity halo exchange
  wallBC_L!   — skip zeroing at MPI-internal left boundaries + halo on L
  measure!    — halo exchange on μ₀, V, μ₁
  update!     — halo exchange on coarse L after restriction
"""
module WaterLilyMPIExt

# Disable precompilation: WaterLily functions are overridden here
__precompile__(false)

using WaterLily
import WaterLily: @loop, measure!, wallBC_L!, exitBC!
using ImplicitGlobalGrid
using MPI
using StaticArrays
using LinearAlgebra: ⋅

# ── Communicator ──────────────────────────────────────────────────────────────

const _comm = Ref{MPI.Comm}(MPI.COMM_WORLD)
WaterLily.set_comm!(comm::MPI.Comm) = (_comm[] = comm)

# ── Global coordinate offset ──────────────────────────────────────────────────

"""
    global_offset(Val(N), T=Float32) → SVector{N,T}

Rank-local origin in global WaterLily index space.
  offset[d] = coords[d] * (nxyz[d] - overlaps[d])  =  coords[d] * nx_loc
"""
function WaterLily.global_offset(::Val{N}, ::Type{T}=Float32) where {N,T}
    g = ImplicitGlobalGrid.global_grid()
    SVector{N,T}(ntuple(d -> T(g.coords[d] * (g.nxyz[d] - g.overlaps[d])), N))
end
WaterLily.global_offset(N::Int, T::Type=Float32) = WaterLily.global_offset(Val(N), T)

# ── Dimension helpers ─────────────────────────────────────────────────────────

# Number of active spatial dimensions (nxyz > 1) in the IGG grid.
_ndims_active() = sum(ImplicitGlobalGrid.global_grid().nxyz .> 1)

# ── Scalar halo exchange (fine grid — via IGG) ───────────────────────────────

function _scalar_halo_igg!(arr::AbstractArray)
    nd = _ndims_active()
    if ndims(arr) < 3
        arr3d = reshape(arr, size(arr)..., ntuple(_->1, 3-ndims(arr))...)
        update_halo!(arr3d; dims=ntuple(identity, nd))
    else
        update_halo!(arr; dims=ntuple(identity, nd))
    end
end

# ── Direct MPI halo exchange (any array size) ────────────────────────────────
#
# IGG pre-allocates MPI send/recv buffers sized for the registered fine grid.
# Calling update_halo! on coarse multigrid arrays produces garbage.  This
# function performs a direct MPI halo exchange using Isend/Irecv! with freshly
# allocated buffers sized for the actual array.  It exchanges 2-cell-wide
# slabs in each active spatial dimension.

# Pre-allocated MPI send/recv buffers keyed by (eltype, slab_shape, dim_tag).
const _mpi_bufs = Dict{Tuple, NTuple{4,Array}}()

function _get_mpi_bufs(::Type{T}, slab_shape::Tuple, dim::Int) where T
    get!(_mpi_bufs, (T, slab_shape, dim)) do
        (zeros(T, slab_shape), zeros(T, slab_shape),
         zeros(T, slab_shape), zeros(T, slab_shape))
    end
end

function _slab(arr::AbstractArray, dim::Int, r::UnitRange)
    # Return a view of arr sliced to indices `r` in dimension `dim`, all others `:`.
    colons = ntuple(i -> i == dim ? r : (:), ndims(arr))
    @view arr[colons...]
end

# Pre-allocated request buffer (max 4 requests per dim exchange)
const _mpi_reqs = MPI.Request[MPI.REQUEST_NULL for _ in 1:4]

function _scalar_halo_mpi!(arr::AbstractArray{T}) where T
    g    = ImplicitGlobalGrid.global_grid()
    nd   = _ndims_active()
    N    = size(arr)
    comm = _comm[]
    for dim in 1:nd
        nleft  = g.neighbors[1, dim]
        nright = g.neighbors[2, dim]
        (nleft < 0 && nright < 0) && continue

        slab_shape = ntuple(i -> i == dim ? 2 : N[i], ndims(arr))
        send_left, recv_left, send_right, recv_right = _get_mpi_bufs(T, slab_shape, dim)

        # Pack send buffers using contiguous slab views
        copyto!(send_left,  _slab(arr, dim, 3:4))
        copyto!(send_right, _slab(arr, dim, N[dim]-3:N[dim]-2))

        # Post all sends/recvs
        nreqs = 0
        if nright >= 0
            nreqs += 1; _mpi_reqs[nreqs] = MPI.Isend(send_right, comm; dest=nright, tag=dim*10)
            nreqs += 1; _mpi_reqs[nreqs] = MPI.Irecv!(recv_right, comm; source=nright, tag=dim*10+1)
        end
        if nleft >= 0
            nreqs += 1; _mpi_reqs[nreqs] = MPI.Isend(send_left, comm; dest=nleft, tag=dim*10+1)
            nreqs += 1; _mpi_reqs[nreqs] = MPI.Irecv!(recv_left, comm; source=nleft, tag=dim*10)
        end
        MPI.Waitall(MPI.RequestSet(_mpi_reqs[1:nreqs]))

        # Unpack recv buffers
        if nleft >= 0
            copyto!(_slab(arr, dim, 1:2), recv_left)
        end
        if nright >= 0
            copyto!(_slab(arr, dim, N[dim]-1:N[dim]), recv_right)
        end
    end
end

# ── Unified scalar halo exchange ─────────────────────────────────────────────

function _is_fine(arr::AbstractArray)
    g  = ImplicitGlobalGrid.global_grid()
    nd = _ndims_active()
    size(arr)[1:nd] == Tuple(g.nxyz[1:nd])
end

function _scalar_halo!(arr::AbstractArray)
    if _is_fine(arr)
        _scalar_halo_igg!(arr)
    else
        _scalar_halo_mpi!(arr)
    end
end

# ── Vector (velocity-shaped) halo exchange ────────────────────────────────────
#
# Pre-allocated buffers keyed by (eltype, spatial_dims) to avoid allocating
# a temporary contiguous array on every call.  After the first time step,
# these are reused and the halo exchange is allocation-free.

const _halo_bufs = Dict{Tuple, Array}()

function _get_halo_buf(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    get!(() -> Array{T}(undef, dims), _halo_bufs, (T, dims))
end

function _velocity_halo!(u::AbstractArray{T,N}) where {T,N}
    D   = size(u, N)                    # number of components (last dim)
    sp  = ntuple(_ -> :, N-1)           # all spatial dims as Colons
    sdims = size(u)[1:N-1]              # spatial dimensions
    tmp = _get_halo_buf(T, sdims)       # single pre-allocated buffer
    for d in 1:D
        copyto!(tmp, @view u[sp..., d])
        _scalar_halo!(tmp)
        copyto!(@view(u[sp..., d]), tmp)
    end
end

# ── Global reduction / halo hooks ────────────────────────────────────────────

WaterLily.global_dot(a, b)    = MPI.Allreduce(a ⋅ b, MPI.SUM, _comm[])
WaterLily.global_sum(a)       = MPI.Allreduce(sum(a), MPI.SUM, _comm[])
WaterLily.global_length(r)    = MPI.Allreduce(length(r), MPI.SUM, _comm[])
WaterLily.global_min(a, b)    = MPI.Allreduce(min(a, b), MPI.MIN, _comm[])
WaterLily.scalar_halo!(x)     = _scalar_halo!(x)
WaterLily.velocity_halo!(u)   = _velocity_halo!(u)

# ── MPI-aware measure! ────────────────────────────────────────────────────────

function WaterLily.measure!(::WaterLily.NoBody, bc::WaterLily.ParallelBC; kwargs...)
    _velocity_halo!(bc.μ₀)
end

function WaterLily.measure!(body::WaterLily.AbstractBody, bc::WaterLily.ParallelBC;
                              t=zero(eltype(bc.σ)), ϵ=1, kwargs...)
    invoke(WaterLily.measure!,
           Tuple{WaterLily.AbstractBody, WaterLily.AbstractBC},
           body, bc; t, ϵ, kwargs...)
    _velocity_halo!(bc.μ₀)
    _velocity_halo!(bc.V)
    nd = _ndims_active()
    μ₁_flat = reshape(bc.μ₁, size(bc.μ₁)[1:nd]..., :)
    _velocity_halo!(μ₁_flat)
end

# ── MPI-aware BC! ─────────────────────────────────────────────────────────────

function WaterLily.BC!(u, bc::WaterLily.ParallelBC, t=0)
    WaterLily.BC!(u, bc.uBC, bc.exitBC, bc.perdir, t)
    _velocity_halo!(u)
end

# ── MPI-aware exitBC! ────────────────────────────────────────────────────────
#
# Serial exitBC! applies convective exit at local N[1]-1 and computes inflow
# from local index 3.  In MPI mode only the rightmost-x rank has the real exit,
# and only the leftmost-x rank has the real inflow.  Global reductions are needed
# for the mean inflow velocity U and the mass-flux correction.

function WaterLily.exitBC!(u, u⁰, Δt, ::WaterLily.ParallelBC)
    g    = ImplicitGlobalGrid.global_grid()
    comm = _comm[]
    N, _ = WaterLily.size_u(u)

    is_inflow  = g.neighbors[1, 1] < 0
    is_exit    = g.neighbors[2, 1] < 0

    # All ranks participate in Allreduce for exit face area
    local_exit_len = is_exit ? length(WaterLily.slice(N .- 2, N[1] - 1, 1, 3)) : 0
    global_exit_len = MPI.Allreduce(local_exit_len, MPI.SUM, comm)

    # All ranks participate in Allreduce for mean inflow velocity
    local_inflow_sum = is_inflow ? sum(@view(u[WaterLily.slice(N .- 2, 3, 1, 3), 1])) : zero(eltype(u))
    U = MPI.Allreduce(local_inflow_sum, MPI.SUM, comm) / global_exit_len

    # Convective exit on rightmost-x ranks only
    if is_exit
        exitR = WaterLily.slice(N .- 2, N[1] - 1, 1, 3)
        @loop u[I, 1] = u⁰[I, 1] - U * Δt * (u⁰[I, 1] - u⁰[I - WaterLily.δ(1, I), 1]) over I ∈ exitR
    end

    # All ranks participate in Allreduce for mass flux correction
    local_exit_sum = is_exit ? sum(@view(u[WaterLily.slice(N .- 2, N[1] - 1, 1, 3), 1])) : zero(eltype(u))
    global_exit_sum = MPI.Allreduce(local_exit_sum, MPI.SUM, comm)
    ∮u = global_exit_sum / global_exit_len - U
    if is_exit
        exitR = WaterLily.slice(N .- 2, N[1] - 1, 1, 3)
        @loop u[I, 1] -= ∮u over I ∈ exitR
    end

    _velocity_halo!(u)
end

# ── MPI-aware wallBC_L! ──────────────────────────────────────────────────────

function WaterLily.wallBC_L!(L, perdir=())
    g  = ImplicitGlobalGrid.global_grid()
    nd = _ndims_active()
    N, n = WaterLily.size_u(L)
    for j in 1:n
        j in perdir && continue
        g.neighbors[1, j] >= 0 && continue   # MPI-internal left: skip zeroing
        @loop L[I,j] = zero(eltype(L)) over I ∈ WaterLily.slice(N, 3, j)
    end
    _velocity_halo!(L)
end

# ── MPI-aware update! (MultiLevelPoisson) ────────────────────────────────────

function WaterLily.update!(ml::WaterLily.MultiLevelPoisson)
    WaterLily.update!(ml.levels[1])
    for l in 2:length(ml.levels)
        WaterLily.restrictL!(ml.levels[l].L, ml.levels[l-1].L,
                             perdir=ml.levels[l-1].perdir)
        _velocity_halo!(ml.levels[l].L)
        WaterLily.update!(ml.levels[l])
    end
end

# ── MPI-aware divisible ───────────────────────────────────────────────────────

WaterLily.divisible(N::Integer) = mod(N,2)==0 && N>8

end # module WaterLilyMPIExt

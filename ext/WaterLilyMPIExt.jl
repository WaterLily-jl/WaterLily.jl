"""
WaterLilyMPIExt — Julia package extension
==========================================
Activated automatically when ImplicitGlobalGrid and MPI are loaded alongside
WaterLily.  Provides MPI-aware overrides for global reductions, halo exchange,
and boundary conditions at MPI-subdomain interfaces.

Uses the `AbstractParMode` dispatch pattern: serial WaterLily dispatches all
hooks through `par_mode[]` (defaults to `Serial()`).  This extension defines
`Parallel <: AbstractParMode` and adds new dispatch methods — no method
overwriting, so precompilation works normally.

Functions with MPI-specific behavior (via dispatch on `::Parallel`):
  _pressureBC!  — zero L at physical walls only (skip MPI-internal) + halo on L
  _exitBC!    — global reductions for inflow/outflow mass flux
  _divisible  — same coarsening threshold as serial (N>4)

Halo exchange uses a cached `_has_neighbors` flag to skip all exchange
when no MPI neighbors exist (e.g. np=1 non-periodic), eliminating the
overhead of IGG's `update_halo!` and buffer copies in that case.
"""
module WaterLilyMPIExt

using WaterLily
import WaterLily: @loop
using ImplicitGlobalGrid
using MPI
using StaticArrays

# ── MPI parallel mode ────────────────────────────────────────────────────────

struct Parallel <: WaterLily.AbstractParMode
    comm::MPI.Comm
    rank::Int
end

_comm() = (WaterLily.par_mode[]::Parallel).comm

WaterLily._mpi_rank(p::Parallel) = p.rank
WaterLily._mpi_comm(p::Parallel) = p.comm

# ── Global coordinate offset ──────────────────────────────────────────────────

"""
    _global_offset(Val(N), T, ::Parallel) → SVector{N,T}

Rank-local origin in global WaterLily index space.
  offset[d] = coords[d] * (nxyz[d] - overlaps[d])  =  coords[d] * nx_loc
"""
function WaterLily._global_offset(::Val{N}, ::Type{T}, ::Parallel) where {N,T}
    g = ImplicitGlobalGrid.global_grid()
    SVector{N,T}(ntuple(d -> T(g.coords[d] * (g.nxyz[d] - g.overlaps[d])), N))
end

# MPI-aware @loop auto-offset: returns SVector sized for the active spatial dims
WaterLily._loop_offset(::Type{T}, p::Parallel) where T =
    WaterLily._global_offset(Val(_ndims_active()), T, p)

# ── MPI initialization ───────────────────────────────────────────────────────

"""
    init_waterlily_mpi(global_dims; perdir=()) → (local_dims, rank, comm)

Initialize MPI domain decomposition for WaterLily.

1. Determines the optimal MPI topology via `MPI.Dims_create`
2. Computes local subdomain dimensions (`global_dims .÷ topology`)
3. Initializes ImplicitGlobalGrid with the correct overlaps and halowidths
4. Sets `par_mode[] = Parallel(comm)` for dispatch-based MPI hooks

Returns `(local_dims::NTuple{N,Int}, rank::Int, comm::MPI.Comm)`.
"""
function WaterLily.init_waterlily_mpi(global_dims::NTuple{N}; perdir=()) where N
    MPI.Initialized() || MPI.Init()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)

    # Optimal MPI topology for N active dimensions
    mpi_dims = Tuple(Int.(MPI.Dims_create(nprocs, zeros(Int, N))))

    # Local interior dims
    local_dims = global_dims .÷ mpi_dims
    all(global_dims .== local_dims .* mpi_dims) ||
        error("Global dims $global_dims not evenly divisible by MPI topology " *
              "$mpi_dims with $nprocs ranks")

    # Pad to 3D for IGG (which always expects 3 dimensions)
    igg_local = ntuple(d -> d <= N ? local_dims[d] + 2 : 1, 3)
    igg_mpi   = ntuple(d -> d <= N ? mpi_dims[d] : 1, 3)
    igg_per   = ntuple(d -> d <= N && d in perdir ? 1 : 0, 3)

    me, dims, np, coords, comm = init_global_grid(
        igg_local...;
        dimx = igg_mpi[1], dimy = igg_mpi[2], dimz = igg_mpi[3],
        overlaps = (2, 2, 2),
        halowidths = (1, 1, 1),
        periodx = igg_per[1], periody = igg_per[2], periodz = igg_per[3],
        init_MPI = false,
    )

    WaterLily.par_mode[] = Parallel(comm, Int(me))
    _init_has_neighbors!()

    if me == 0
        topo = join(string.(dims[1:N]), "×")
        loc  = join(string.(local_dims), "×")
        glob = join(string.(global_dims), "×")
        @info "WaterLily MPI: $(np) ranks, topology=$(topo), " *
              "local=$(loc), global=$(glob)"
    end

    return local_dims, me, comm
end

# ── Dimension helpers ─────────────────────────────────────────────────────────

# Number of active spatial dimensions (nxyz > 1) in the IGG grid.
_ndims_active() = sum(ImplicitGlobalGrid.global_grid().nxyz .> 1)

# True if any MPI neighbor exists in any active dimension (cached after init).
const _has_neighbors = Ref(false)
function _init_has_neighbors!()
    g = ImplicitGlobalGrid.global_grid()
    nd = _ndims_active()
    _has_neighbors[] = any(g.neighbors[s, d] >= 0 for s in 1:2, d in 1:nd)
end

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
    colons = ntuple(i -> i == dim ? r : (:), ndims(arr))
    @view arr[colons...]
end

# Pre-allocated request buffer (max 4 requests per dim exchange)
const _mpi_reqs = MPI.Request[MPI.REQUEST_NULL for _ in 1:4]

function _scalar_halo_mpi!(arr::AbstractArray{T}) where T
    g    = ImplicitGlobalGrid.global_grid()
    nd   = _ndims_active()
    N    = size(arr)
    comm = _comm()
    for dim in 1:nd
        nleft  = g.neighbors[1, dim]
        nright = g.neighbors[2, dim]
        (nleft < 0 && nright < 0) && continue

        # halowidth=1: 1-cell-wide slabs
        slab_shape = ntuple(i -> i == dim ? 1 : N[i], ndims(arr))
        send_left, recv_left, send_right, recv_right = _get_mpi_bufs(T, slab_shape, dim)

        # Pack send buffers: first/last interior cells (index 2 and N-1)
        copyto!(send_left,  _slab(arr, dim, 2:2))
        copyto!(send_right, _slab(arr, dim, N[dim]-1:N[dim]-1))

        # Post all non-blocking sends/recvs, then Waitall for max overlap
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

        # Unpack recv buffers: into ghost cells (index 1 and N)
        if nleft >= 0
            copyto!(_slab(arr, dim, 1:1), recv_left)
        end
        if nright >= 0
            copyto!(_slab(arr, dim, N[dim]:N[dim]), recv_right)
        end
    end
end

# ── Unified scalar halo exchange ─────────────────────────────────────────────

function _is_fine(arr::AbstractArray)
    g  = ImplicitGlobalGrid.global_grid()
    nd = _ndims_active()
    size(arr)[1:nd] == Tuple(g.nxyz[1:nd])
end

function _do_scalar_halo!(arr::AbstractArray)
    _has_neighbors[] || return
    if _is_fine(arr)
        _scalar_halo_igg!(arr)
    else
        _scalar_halo_mpi!(arr)
    end
end

# ── Vector (velocity-shaped) halo exchange ────────────────────────────────────

const _halo_bufs = Dict{Tuple, Array}()

function _get_halo_buf(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    get!(() -> Array{T}(undef, dims), _halo_bufs, (T, dims))
end

function _do_velocity_halo!(u::AbstractArray{T,N}) where {T,N}
    _has_neighbors[] || return
    D   = size(u, N)                    # number of components (last dim)
    sp  = ntuple(_ -> :, N-1)           # all spatial dims as Colons
    sdims = size(u)[1:N-1]              # spatial dimensions
    tmp = _get_halo_buf(T, sdims)       # single pre-allocated buffer
    for d in 1:D
        copyto!(tmp, @view u[sp..., d])
        _do_scalar_halo!(tmp)
        copyto!(@view(u[sp..., d]), tmp)
    end
end

# ── Dispatch hooks for Parallel ──────────────────────────────────────────────
WaterLily._global_allreduce(x, ::Parallel)        = MPI.Allreduce(x, MPI.SUM, _comm())
WaterLily._global_min(a, b, ::Parallel)           = MPI.Allreduce(min(a, b), MPI.MIN, _comm())
WaterLily._global_max(x, ::Parallel)              = MPI.Allreduce(x, MPI.MAX, _comm())
WaterLily._scalar_halo!(x, ::Parallel)            = _do_scalar_halo!(x)
WaterLily._velocity_halo!(u, ::Parallel)          = _do_velocity_halo!(u)

# Communication hooks: in parallel, MPI halo handles periodicity
WaterLily._comm!(a, perdir, ::Parallel)            = _do_scalar_halo!(a)
WaterLily._velocity_comm!(a, perdir, ::Parallel)   = _do_velocity_halo!(a)

# ── MPI-aware exitBC! ────────────────────────────────────────────────────────

function WaterLily._exitBC!(u, u⁰, Δt, ::Parallel)
    g    = ImplicitGlobalGrid.global_grid()
    comm = _comm()
    N, _ = WaterLily.size_u(u)

    is_inflow  = g.neighbors[1, 1] < 0
    is_exit    = g.neighbors[2, 1] < 0

    # All ranks participate in Allreduce for exit face area
    local_exit_len = is_exit ? length(WaterLily.slice(N .- 1, N[1], 1, 2)) : 0
    global_exit_len = MPI.Allreduce(local_exit_len, MPI.SUM, comm)

    # All ranks participate in Allreduce for mean inflow velocity
    local_inflow_sum = is_inflow ? sum(@view(u[WaterLily.slice(N .- 1, 2, 1, 2), 1])) : zero(eltype(u))
    U = MPI.Allreduce(local_inflow_sum, MPI.SUM, comm) / global_exit_len

    # Convective exit on rightmost-x ranks only
    if is_exit
        exitR = WaterLily.slice(N .- 1, N[1], 1, 2)
        @loop u[I, 1] = u⁰[I, 1] - U * Δt * (u⁰[I, 1] - u⁰[I - WaterLily.δ(1, I), 1]) over I ∈ exitR
    end

    # All ranks participate in Allreduce for mass flux correction
    local_exit_sum = is_exit ? sum(@view(u[WaterLily.slice(N .- 1, N[1], 1, 2), 1])) : zero(eltype(u))
    global_exit_sum = MPI.Allreduce(local_exit_sum, MPI.SUM, comm)
    ∮u = global_exit_sum / global_exit_len - U
    if is_exit
        exitR = WaterLily.slice(N .- 1, N[1], 1, 2)
        @loop u[I, 1] -= ∮u over I ∈ exitR
    end

    _do_velocity_halo!(u)
end

# ── MPI-aware pressureBC! ────────────────────────────────────────────────────

function WaterLily._pressureBC!(L, perdir, ::Parallel)
    # master-style layout: BC! on μ₀ already zeros L at wall faces; just sync halo
    _do_velocity_halo!(L)
end

# ── MPI-aware BC! ────────────────────────────────────────────────────────────
# At rank-internal boundaries, skip Dirichlet writes so halo exchange can
# provide neighbor's interior data.  Only write BC at physical walls.
function WaterLily._BC!(a, uBC::Function, saveexit, perdir, t, ::Parallel)
    g  = ImplicitGlobalGrid.global_grid()
    N, n = WaterLily.size_u(a)
    for i ∈ 1:n, j ∈ 1:n
        j in perdir && continue
        phys_left  = g.neighbors[1, j] < 0
        phys_right = g.neighbors[2, j] < 0
        if i==j # Normal direction, Dirichlet — only at physical walls
            if phys_left
                for s ∈ (1,2)
                    @loop a[I,i] = uBC(i,WaterLily.loc(i,I),t) over I ∈ WaterLily.slice(N,s,j)
                end
            end
            if phys_right && (!saveexit || i>1)
                @loop a[I,i] = uBC(i,WaterLily.loc(i,I),t) over I ∈ WaterLily.slice(N,N[j],j)
            end
        else    # Tangential Neumann mirror — only at physical walls (halo handles rank-internal)
            if phys_left
                @loop a[I,i] = uBC(i,WaterLily.loc(i,I),t)+a[I+WaterLily.δ(j,I),i]-uBC(i,WaterLily.loc(i,I+WaterLily.δ(j,I)),t) over I ∈ WaterLily.slice(N,1,j)
            end
            if phys_right
                @loop a[I,i] = uBC(i,WaterLily.loc(i,I),t)+a[I-WaterLily.δ(j,I),i]-uBC(i,WaterLily.loc(i,I-WaterLily.δ(j,I)),t) over I ∈ WaterLily.slice(N,N[j],j)
            end
        end
    end
    WaterLily.velocity_comm!(a, perdir)
end

# ── MPI-aware divisible ───────────────────────────────────────────────────────
# Same threshold as serial (N>4). Coarse-level comm cost is negligible thanks
# to `_has_neighbors` short-circuiting (no exchange when no MPI neighbors exist)
# and tiny array sizes at the coarsest levels.

WaterLily._divisible(N, ::Parallel) = mod(N,2)==0 && N>4

# Coarsest solve: use PCG so global dot-products catch the null-space mode
# that a local red-black smoother cannot reach (MPI loses one V-cycle level).
# Combined with per-V-cycle `pin_pressure!` in `solver!` to keep `p.x`'s
# mean bounded between iterations.
WaterLily._coarsest_solve!(p, ω, ::Parallel) = WaterLily.pcg!(p; it=32)

end # module WaterLilyMPIExt

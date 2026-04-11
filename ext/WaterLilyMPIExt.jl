"""
WaterLilyMPIExt — Julia package extension
==========================================
Activated automatically when ImplicitGlobalGrid and MPI are loaded alongside
WaterLily.  Provides MPI-aware overrides for global reductions, halo exchange,
and boundary conditions at MPI-subdomain interfaces.

The serial WaterLily code uses hook functions (global_dot, global_sum,
global_length, global_min, scalar_halo!, velocity_halo!) that are no-ops
in serial.  This extension overrides them with MPI.Allreduce and halo
exchange.  The serial BC!, measure!, and Poisson solver
already call these hooks, so no MPI-specific overrides are needed for
them.

Functions still overridden here (beyond hooks):
  wallBC_L!   — zero L at physical walls only (skip MPI-internal) + halo on L
  exitBC!     — global reductions for inflow/outflow mass flux
  divisible   — stricter coarsening threshold (N>8) for multigrid
"""
module WaterLilyMPIExt

# Disable precompilation: WaterLily functions are overridden here
__precompile__(false)

using WaterLily
import WaterLily: @loop, wallBC_L!, exitBC!
using ImplicitGlobalGrid
using MPI
using StaticArrays

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

# ── MPI initialization ───────────────────────────────────────────────────────

"""
    init_waterlily_mpi(global_dims; perdir=()) → (local_dims, rank, comm)

Initialize MPI domain decomposition for WaterLily.

1. Determines the optimal MPI topology via `MPI.Dims_create`
2. Computes local subdomain dimensions (`global_dims .÷ topology`)
3. Initializes ImplicitGlobalGrid with the correct overlaps and halowidths
4. Sets the MPI communicator for WaterLily's global reductions

Returns `(local_dims::NTuple{N,Int}, rank::Int, comm::MPI.Comm)`.

`AutoBody` automatically applies the global coordinate offset and the serial
hook functions (velocity_halo!, scalar_halo!, global_dot, etc.) are overridden
by this extension, so parallel user scripts need only differ from serial by
calling this function and using `local_dims` in the `Simulation` constructor.
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
    igg_local = ntuple(d -> d <= N ? local_dims[d] + 4 : 1, 3)
    igg_mpi   = ntuple(d -> d <= N ? mpi_dims[d] : 1, 3)
    igg_per   = ntuple(d -> d <= N && d in perdir ? 1 : 0, 3)

    me, dims, np, coords, comm = init_global_grid(
        igg_local...;
        dimx = igg_mpi[1], dimy = igg_mpi[2], dimz = igg_mpi[3],
        overlaps = (4, 4, 4),
        halowidths = (2, 2, 2),
        periodx = igg_per[1], periody = igg_per[2], periodz = igg_per[3],
        init_MPI = false,
    )

    _comm[] = comm

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

WaterLily.global_dot(a, b; R=inside(a)) = MPI.Allreduce(WaterLily.dot_R(a, b, R), MPI.SUM, _comm[])
WaterLily.global_sum(a; R=inside(a)) = MPI.Allreduce(WaterLily.sum_R(a, R), MPI.SUM, _comm[])
WaterLily.global_length(r) = MPI.Allreduce(length(r), MPI.SUM, _comm[])
WaterLily.global_min(a, b) = MPI.Allreduce(min(a, b), MPI.MIN, _comm[])
WaterLily.scalar_halo!(x) = _scalar_halo!(x)
WaterLily.velocity_halo!(u) = _velocity_halo!(u)

# ── MPI-aware exitBC! ────────────────────────────────────────────────────────
#
# Serial exitBC! applies convective exit at local N[1]-1 and computes inflow
# from local index 3.  In MPI mode only the rightmost-x rank has the real exit,
# and only the leftmost-x rank has the real inflow.  Global reductions are needed
# for the mean inflow velocity U and the mass-flux correction.

function WaterLily.exitBC!(u, u⁰, Δt, ::WaterLily.BC)
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
    N, n = WaterLily.size_u(L)
    for j in 1:n
        j in perdir && continue
        if g.neighbors[1, j] < 0  # physical left wall
            @loop L[I,j] = zero(eltype(L)) over I ∈ WaterLily.slice(N, 3, j)
        end
        if g.neighbors[2, j] < 0  # physical right wall
            @loop L[I,j] = zero(eltype(L)) over I ∈ WaterLily.slice(N, N[j]-1, j)
        end
    end
    _velocity_halo!(L)
end

# ── MPI-aware divisible ───────────────────────────────────────────────────────

WaterLily.divisible(N::Integer) = mod(N,2)==0 && N>8

end # module WaterLilyMPIExt

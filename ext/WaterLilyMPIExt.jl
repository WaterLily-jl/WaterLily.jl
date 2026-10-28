"""
WaterLilyMPIExt — activated when ImplicitGlobalGrid and MPI are loaded with
WaterLily. Adds `Parallel <: AbstractParMode` methods for the `par_mode[]`
dispatch hooks (reductions, halo exchange, BC gates) — new methods only, no
overwriting, so precompilation works.

Topology queries read IGG's `global_grid()` directly (single source of truth;
costs a few ns, invisible under the ~50 ns `par_mode[]` dynamic dispatch).
The only stored state is a genuine cache: the halo plans (regions, buffers,
requests) per array type and size.
"""
module WaterLilyMPIExt

using WaterLily
import WaterLily: @loop
using ImplicitGlobalGrid
using MPI
using StaticArrays
import KernelAbstractions
using ForwardDiff: Dual, Partials, value, partials

# ── MPI parallel mode ────────────────────────────────────────────────────────

struct Parallel <: WaterLily.AbstractParMode
    comm::MPI.Comm
    rank::Int
end

_comm() = (WaterLily.par_mode[]::Parallel).comm

WaterLily._mpi_rank(p::Parallel)   = p.rank
WaterLily._mpi_comm(p::Parallel)   = p.comm
WaterLily._mpi_nprocs(p::Parallel) = MPI.Comm_size(p.comm)

# ── Live-grid accessors ──────────────────────────────────────────────────────
# `par_mode[] isa Parallel` implies `init_waterlily_mpi` ran, so the grid is
# initialized; after `finalize_global_grid()` these fail loudly via IGG's
# initialization check rather than serving stale topology.

const _grid = ImplicitGlobalGrid.global_grid           # the active GlobalGrid

_nd() = count(>(1), _grid().nxyz)                      # active spatial dims

# Halo exchange only covers dims split across ranks. Periodic dims owned in
# full by every rank wrap locally via `perBC!` (exactly as in serial) — going
# through MPI for the self-wrap costs the full per-call floor for nothing.
_any_decomposed() = (g = _grid(); any(g.dims[d] > 1 for d in 1:_nd()))

# Periodic dims that are NOT decomposed — handled by a local `perBC!` copy
_local_perdir(perdir) = (g = _grid(); filter(j -> g.dims[j] == 1, perdir))

# ── Global coordinate offset ──────────────────────────────────────────────────

# Rank-local origin in global index space: offset[d] = coords[d] * nx_loc.
function WaterLily._global_offset(::Val{N}, ::Type{T}, ::Parallel) where {N,T}
    g = _grid()
    SVector{N,T}(ntuple(d -> T(g.coords[d] * (g.nxyz[d] - g.overlaps[d])), N))
end

# @loop auto-offset. Float and Dual eltypes keep T, so offsets carry their
# precision / AD tags; any other first argument (Bool or Int arrays, arrays of
# SVectors, a tuple of arrays) takes `loc`'s default Float32.
WaterLily._loop_offset(::Type{T}, p::Parallel) where T<:Union{AbstractFloat,Dual} =
    WaterLily._global_offset(Val(_nd()), T, p)
WaterLily._loop_offset(::Type, p::Parallel) =
    WaterLily._global_offset(Val(_nd()), Float32, p)

# A body's `bbox` is in global cell indices: shift the block there and the result
# back, clipped to the block (an empty range on ranks the box misses).
function WaterLily._local_bbox(body, R::CartesianIndices{N}, p::Parallel) where N
    o = CartesianIndex(Tuple(WaterLily._global_offset(Val(N), Int, p)))
    (WaterLily.bbox(body, R .+ o) .- o) ∩ R
end

# ── MPI initialization ───────────────────────────────────────────────────────

# Rank layout minimising the per-rank halo surface Σ_d ∏_{k≠d} local[k], with
# topo[d] dividing global_dims[d]. Seeded by `MPI.Dims_create`, which wins
# ties but is suboptimal on anisotropic grids (1024×64 np=8: (8,1) beats its
# (4,2) by 33% less surface).
function _shape_aware_topology(global_dims::NTuple{N,Int}, nprocs::Int) where N
    best_topo = Tuple(Int.(MPI.Dims_create(nprocs, zeros(Int, N))))
    best_score = if all(global_dims[d] % best_topo[d] == 0 for d in 1:N)
        _halo_surface(global_dims, best_topo)
    else
        typemax(Int)
    end
    # all N-tuples of divisors with product == nprocs
    function visit(remaining::Int, partial::Tuple)
        if length(partial) == N - 1
            topo = (partial..., remaining)
            all(global_dims[d] % topo[d] == 0 for d in 1:N) || return
            s = _halo_surface(global_dims, topo)
            if s < best_score                 # strict — `Dims_create` wins ties
                best_score = s
                best_topo  = topo
            end
            return
        end
        for d in 1:remaining
            remaining % d == 0 && visit(remaining ÷ d, (partial..., d))
        end
    end
    visit(nprocs, ())
    best_score == typemax(Int) &&
        error("No valid MPI topology for global_dims=$global_dims with nprocs=$nprocs")
    return best_topo
end

@inline function _halo_surface(global_dims::NTuple{N,Int}, topo::NTuple{N,Int}) where N
    s = 0
    local_dims = ntuple(d -> global_dims[d] ÷ topo[d], N)
    for d in 1:N
        face = 1
        for k in 1:N
            k == d && continue
            face *= local_dims[k]
        end
        s += face
    end
    return s
end

"""
    init_waterlily_mpi(global_dims; perdir=(), select_device=!_gpu_bound()) → (local_dims, rank, comm)

Initialize MPI domain decomposition: picks the rank topology
(`_shape_aware_topology`), initializes ImplicitGlobalGrid, and sets
`par_mode[] = Parallel(comm, rank)`. With CUDA or AMDGPU loaded,
`select_device=true` gives each node-local rank its own GPU (ImplicitGlobalGrid's
`select_device`); when the launcher already binds one GPU per rank (e.g. Slurm
`--gpus-per-task=1`), each rank keeps the GPU it sees.
"""
function WaterLily.init_waterlily_mpi(global_dims::NTuple{N}; perdir=(), select_device=!_gpu_bound()) where N
    MPI.Initialized() || MPI.Init()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)

    mpi_dims   = _shape_aware_topology(global_dims, nprocs)
    local_dims = global_dims .÷ mpi_dims
    all(global_dims .== local_dims .* mpi_dims) ||
        error("Global dims $global_dims not evenly divisible by MPI topology " *
              "$mpi_dims with $nprocs ranks")

    # IGG always expects 3 dimensions — pad
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
        select_device,
    )

    WaterLily.par_mode[] = Parallel(comm, Int(me))
    empty!(_halos)  # the plans of any previous grid (re-init in the same session)
    _overlap_min[] = parse(Int, get(ENV, "WATERLILY_OVERLAP_MIN", string(2^20)))
    _aware_min[] = (n = parse(Int, get(ENV, "WATERLILY_GPU_AWARE_MPI", "0")); n == 0 ? typemax(Int) : n)

    if me == 0
        topo = join(string.(dims[1:N]), "×")
        loc  = join(string.(local_dims), "×")
        glob = join(string.(global_dims), "×")
        @info "WaterLily MPI: $(np) ranks, topology=$(topo), " *
              "local=$(loc), global=$(glob)"
    end

    return local_dims, me, comm
end

# One visible GPU per process: the launcher bound the GPUs (ImplicitGlobalGrid's
# select_device would count one GPU for all the ranks of a node and stop)
_gpu_bound() = any(("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")) do k
    v = get(ENV, k, ""); !isempty(v) && !occursin(',', v)
end

# ── Halo exchange ─────────────────────────────────────────────────────────────
# One message per neighbour rank, the diagonal ones included (up to 3^nd - 1), so
# that the edge and corner ghosts come straight from the rank that owns them: the
# ghosts of exchanging one dim after the other, in one round of messages. One
# kernel packs every region into a flat send buffer, the messages all go at once,
# and one kernel unpacks the flat receive buffer. A plan (regions, buffers,
# requests) is cached per array type and size: scalar, velocity, coarse-level,
# Dual and Bool arrays all take this path, on every backend. GPU halos go through
# host memory, or device to device with GPU-aware MPI: WATERLILY_GPU_AWARE_MPI=0
# (or unset) never, 1 always, N > 1 for the halos of at least N bytes.

const _halos = Dict{Tuple, Any}()
const _overlap_min = Ref(2^20)         # the fewest cells of a loop that `comm!(f, ...)` splits
const _aware_min   = Ref(typemax(Int))  # the fewest bytes of a halo passed to MPI in device memory

# The rank in direction `d` (-1, 0 or 1 per spatial dim), or -1 past a physical boundary
function _peer(d)
    g = _grid(); c = [g.coords[k] + get(d, k, 0) for k in 1:3]
    all(k -> g.periods[k] == 1 || 0 <= c[k] < g.dims[k], 1:3) ? MPI.Cart_rank(_comm(), c) : -1
end

# The cells of `a` sent toward (`send`) or received from direction `d`: in a dim of
# `d`, the first or last interior layer, or the ghost layer; in the other decomposed
# dims, the interior and the ghosts of a physical side (they travel with the
# messages, as when the dims go one after the other); in the rest, every cell.
_halo_cells(a, d, send) = CartesianIndices(ntuple(ndims(a)) do k
    n, dk = size(a, k), get(d, k, 0)
    dk < 0 ? (send ? (2:2) : (1:1)) : dk > 0 ? (send ? (n-1:n-1) : (n:n)) :
    WaterLily.decomposed(k) ? ((WaterLily.phys_left(k) ? 1 : 2):(WaterLily.phys_right(k) ? n : n-1)) : (1:n)
end)
_dirkey(d) = evalpoly(3, d .+ 1)

function _halo_plan(a::AbstractArray{T}) where T
    dirs = [d for d in Iterators.product(ntuple(k -> WaterLily.decomposed(k) ? (-1:1) : (0:0), _nd())...)
            if any(!=(0), d) && _peer(d) >= 0]
    sort!(dirs, by=d -> count(!=(0), d))  # faces first: most threads of the kernels find their region at once
    peers = unique(_peer.(dirs))
    msgs = [filter(d -> _peer(d) == p, dirs) for p in peers]  # the directions of the message to/from each peer
    # a message holds its regions in the order of the sender's directions
    S = Tuple(_halo_cells(a, d, true) for m in msgs for d in sort(m, by=_dirkey))
    R = Tuple(_halo_cells(a, d, false) for m in msgs for d in sort(m, by=d -> _dirkey(.-d)))
    lens = [sum(d -> length(_halo_cells(a, d, true)), m) for m in msgs]; ends = cumsum(lens)
    buffers(n) = n * sizeof(T) >= _aware_min[] ? (b = similar(a, T, n); (b, b)) : WaterLily._host_buffers(a, T, n)
    (s, sh), (r, rh) = buffers(ends[end]), buffers(ends[end])
    (; S=WaterLily.chain(S), R=WaterLily.chain(R), s, sh, r, rh, peers, ranges=UnitRange.(ends .- lens .+ 1, ends),
       sreq=MPI.UnsafeMultiRequest(length(peers)), rreq=MPI.UnsafeMultiRequest(length(peers)))
end

# The halo of `a`, with `work()` run while its messages travel. One halo at a time:
# a plan's buffers serve every array of its type and size.
function _halo!(work, a)
    _any_decomposed() || return work()
    _halo!(work, a, get!(() -> _halo_plan(a), _halos, (typeof(a), size(a))))
end
function _halo!(work, a, h)
    WaterLily.pack!(h.s, a, h.S)
    _sync(a)  # the pack is done, and so is the last unpack from h.r, which MPI.Irecv! may now overwrite
    for (i, p) in enumerate(h.peers)
        MPI.Irecv!(view(h.rh, h.ranges[i]), _comm(), h.rreq[i]; source=p, tag=0)
    end
    for (i, p) in enumerate(h.peers)
        MPI.Isend(view(h.sh, h.ranges[i]), _comm(), h.sreq[i]; dest=p, tag=0)
    end
    work()
    MPI.Waitall(h.rreq)
    WaterLily.unpack!(a, h.r, h.R)  # not waited for: later kernels on the backend's queue run after it
    MPI.Waitall(h.sreq)
end
_halo!(a) = _halo!(Returns(nothing), a)

# Kernels may still be writing an array that MPI is about to read: wait for them.
_sync(a) = KernelAbstractions.synchronize(KernelAbstractions.get_backend(a))

# ── ForwardDiff.Dual support ─────────────────────────────────────────────────
# `Dual{T,V,N}` is `isbits` with layout [value, partial₁, …, partialₙ] of `V`.
# Reinterpret to flat `V` for the wire, reconstruct on receive. SUM is
# element-wise on the flat view (derivative is linear); MIN/MAX are not —
# the partials of the extremum aren't the extrema of the partials.

@inline _dual_flat(x::Dual{T,V,N}) where {T,V,N} = V[value(x); partials(x).values...]
@inline _dual_unflat(::Type{Dual{T,V,N}}, f::AbstractVector{V}) where {T,V,N} =
    Dual{T,V,N}(f[1], Partials{N,V}(NTuple{N,V}(@view f[2:end])))

# Zero-copy flat-V alias of a contiguous Dual array. Callers must keep the
# parent rooted (GC.@preserve) for the lifetime of the view. MPI.jl has no
# `Buffer` for a host `ReinterpretArray`, hence the `unsafe_wrap`; `reinterpret`
# of a device array (CuArray, ROCArray) is itself a device array.
@inline function _dual_view(arr::Array{D}) where {Tg,V,N,D<:Dual{Tg,V,N}}
    unsafe_wrap(Array, Ptr{V}(pointer(arr)), length(arr) * (N + 1))
end
@inline _dual_view(arr::AbstractArray{<:Dual{Tg,V}}) where {Tg,V} = reinterpret(V, vec(arr))

_dual_sum(x::Dual{T,V,N}) where {T,V,N} =
    (f = _dual_flat(x); MPI.Allreduce!(f, MPI.SUM, _comm()); _dual_unflat(Dual{T,V,N}, f))

# MIN/MAX: Allgather the values, find the rank holding the global extremum,
# Bcast its full Dual (value + partials).
function _dual_extremum(x::Dual{T,V,N}, op) where {T,V,N}
    comm = _comm()
    vals = MPI.Allgather(value(x), comm)
    root = (op === MPI.MIN ? argmin(vals) : argmax(vals)) - 1
    f = MPI.Comm_rank(comm) == root ? _dual_flat(x) : zeros(V, N + 1)
    MPI.Bcast!(f, root, comm); _dual_unflat(Dual{T,V,N}, f)
end

# ── Dispatch hooks for Parallel ──────────────────────────────────────────────
WaterLily._global_allreduce(x, ::Parallel) = MPI.Allreduce(x, MPI.SUM, _comm())
WaterLily._global_allreduce(x::AbstractArray, p::Parallel) = WaterLily._global_allreduce!(copy(x), p)
WaterLily._global_allreduce(x::Dual, ::Parallel) = _dual_sum(x)
# In place on the array's backend (a Dual array as its flat V alias). A kernel may
# still be writing a device array: wait for it, as before a halo exchange.
WaterLily._global_allreduce!(x::AbstractArray, ::Parallel) = (_sync(x); MPI.Allreduce!(x, MPI.SUM, _comm()); x)
WaterLily._global_allreduce!(x::AbstractArray{<:Dual}, ::Parallel) =
    (_sync(x); GC.@preserve x MPI.Allreduce!(_dual_view(x), MPI.SUM, _comm()); x)
WaterLily._global_min(a, b, ::Parallel) = MPI.Allreduce(min(a, b), MPI.MIN, _comm())
WaterLily._global_min(a::Dual, b::Dual, ::Parallel) = _dual_extremum(min(a, b), MPI.MIN)
WaterLily._global_max(x, ::Parallel) = MPI.Allreduce(x, MPI.MAX, _comm())
WaterLily._global_max(x::Dual, ::Parallel) = _dual_extremum(x, MPI.MAX)
WaterLily._scalar_halo!(x, ::Parallel) = _halo!(x)
WaterLily._velocity_halo!(u, ::Parallel) = _halo!(u)

# Decomposed periodic dims wrap through the halo (IGG periodic topology);
# non-decomposed ones wrap locally via `perBC!`, exactly as in serial.
# `perBC!` runs first so the halo slabs carry the wrapped corner values.
function WaterLily._comm!(a, perdir, ::Parallel)
    WaterLily.perBC!(a, _local_perdir(perdir))
    _halo!(a)
end
WaterLily._comm!(f, a, perdir, R, ::Parallel) = (WaterLily.perBC!(a, _local_perdir(perdir)); _overlap!(f, a, R))
function _velocity_perBC!(a, perdir)
    lp = _local_perdir(perdir)
    isempty(lp) || for i in 1:size(a, ndims(a))
        WaterLily.perBC!(selectdim(a, ndims(a), i), lp)
    end
end
WaterLily._velocity_comm!(a, perdir, ::Parallel) = (_velocity_perBC!(a, perdir); _halo!(a))
WaterLily._velocity_comm!(f, a, perdir, R, ::Parallel) = (_velocity_perBC!(a, perdir); _overlap!(f, a, R))

# `f(R)` with the halo of `a` travelling during `f(inner(R))`, for loops long enough to hide it behind
function _overlap!(f, a, R)
    _any_decomposed() && length(R) >= _overlap_min[] || return (_halo!(a); f(R))
    _halo!(() -> f(WaterLily.inner(R)), a)
    f(WaterLily.shell(R))
end

# Rank-internal faces are non-physical: `_BC!`/`_exitBC!` skip writes there
# and the halo supplies neighbor data.
WaterLily._phys_left(j,  ::Parallel) = _grid().neighbors[1, j] < 0
WaterLily._phys_right(j, ::Parallel) = _grid().neighbors[2, j] < 0

# `effective_perdir` filter: conv_diff!'s periodic stencil `ϕuP` wraps within
# the local array — wrong when the periodic partner is on a remote rank and
# halowidth=1 can't supply its 2-cell stencil. Decomposed dims route to the
# `Val{false}` stencil, whose 1 ghost cell the halo's periodic wrap fills.
WaterLily._decomposed(j, ::Parallel) = j <= _nd() && _grid().dims[j] > 1

end # module WaterLilyMPIExt

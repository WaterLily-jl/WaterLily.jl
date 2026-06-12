"""
WaterLilyMPIExt — activated when ImplicitGlobalGrid and MPI are loaded with
WaterLily. Adds `Parallel <: AbstractParMode` methods for the `par_mode[]`
dispatch hooks (reductions, halo exchange, BC gates) — new methods only, no
overwriting, so precompilation works.

Topology queries read IGG's `global_grid()` directly (single source of truth;
costs a few ns, invisible under the ~50 ns `par_mode[]` dynamic dispatch).
The only stored state is genuine caches: per-size coarse `GlobalGrid`s and
pre-allocated MPI buffers.
"""
module WaterLilyMPIExt

using WaterLily
import WaterLily: @loop
using ImplicitGlobalGrid
using MPI
using StaticArrays
using ForwardDiff: Dual, Partials, value, partials
using EllipsisNotation

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
_decomp_dims() = (g = _grid(); filter(d -> g.dims[d] > 1, ntuple(identity, _nd())))
_any_decomposed() = (g = _grid(); any(g.dims[d] > 1 for d in 1:_nd()))

# Periodic dims that are NOT decomposed — handled by a local `perBC!` copy
_local_perdir(perdir) = (g = _grid(); filter(j -> g.dims[j] == 1, perdir))

# ── Global coordinate offset ──────────────────────────────────────────────────

# Rank-local origin in global index space: offset[d] = coords[d] * nx_loc.
function WaterLily._global_offset(::Val{N}, ::Type{T}, ::Parallel) where {N,T}
    g = _grid()
    SVector{N,T}(ntuple(d -> T(g.coords[d] * (g.nxyz[d] - g.overlaps[d])), N))
end

# @loop auto-offset. Keeps T so Float32/Float64/Dual offsets carry correct
# precision / AD tags; Bool arrays can't hold the offset — divert to Float32.
WaterLily._loop_offset(::Type{T}, p::Parallel) where T =
    WaterLily._global_offset(Val(_nd()), T, p)
WaterLily._loop_offset(::Type{Bool}, p::Parallel) =
    WaterLily._global_offset(Val(_nd()), Float32, p)

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
    init_waterlily_mpi(global_dims; perdir=()) → (local_dims, rank, comm)

Initialize MPI domain decomposition: picks the rank topology
(`_shape_aware_topology`), initializes ImplicitGlobalGrid, and sets
`par_mode[] = Parallel(comm, rank)`.
"""
function WaterLily.init_waterlily_mpi(global_dims::NTuple{N}; perdir=()) where N
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
    )

    WaterLily.par_mode[] = Parallel(comm, Int(me))
    # Drop per-size caches from any previous grid (re-init in the same session)
    foreach(empty!, (_coarse_grids, _mpi_bufs, _halo_bufs, _agg_cache))

    if me == 0
        topo = join(string.(dims[1:N]), "×")
        loc  = join(string.(local_dims), "×")
        glob = join(string.(global_dims), "×")
        @info "WaterLily MPI: $(np) ranks, topology=$(topo), " *
              "local=$(loc), global=$(glob)"
    end

    return local_dims, me, comm
end

# ── Per-size GlobalGrid side-table (coarse multigrid arrays) ─────────────────
# IGG ≥ 0.17: `create_global_grid` builds a `GlobalGrid` without side effects;
# `update_halo!(arr; active_global_grid=gg)` swaps to it for one call. One
# grid per rank-local coarse size (same topology / periods / overlaps as the
# fine grid), shared by all `MultiLevelPoisson`s in the process.

const _coarse_grids = Dict{Tuple, ImplicitGlobalGrid.GlobalGrid}()

function _create_coarse_grid(arr_size::Tuple)
    g    = _grid()
    nxyz = ntuple(d -> d <= _nd() ? arr_size[d] : 1, 3)
    ImplicitGlobalGrid.create_global_grid(
        nxyz...;
        dimx       = g.dims[1],    dimy    = g.dims[2],    dimz    = g.dims[3],
        periodx    = g.periods[1], periody = g.periods[2], periodz = g.periods[3],
        overlaps   = Tuple(g.overlaps),
        halowidths = Tuple(g.halowidths),
        comm       = MPI.COMM_WORLD,   # IGG re-runs Cart_create internally
        quiet      = true,
    )
end

@inline _coarse_grid(arr_size::Tuple) =
    get!(() -> _create_coarse_grid(arr_size), _coarse_grids, arr_size)

@inline function _is_fine_size(sz::Tuple, nd::Int)
    nxyz = _grid().nxyz
    @inbounds for d in 1:nd
        sz[d] != nxyz[d] && return false
    end
    return true
end

# Test / introspection API — the grid `update_halo!` resolves for `arr_size`.
@inline _grid_for(arr_size::Tuple) = _is_fine_size(arr_size, _nd()) ? _grid() : _coarse_grid(arr_size)

# ── Scalar halo exchange (native eltypes — via IGG) ──────────────────────────
# Fine-sized arrays skip the Dict lookup AND the `active_global_grid=` kwarg,
# so IGG's activate/restore dance never fires on the most-called path.

function _scalar_halo_igg!(arr::AbstractArray)
    arr3d = ndims(arr) < 3 ?
            reshape(arr, size(arr)..., ntuple(_ -> 1, 3 - ndims(arr))...) : arr
    if _is_fine_size(size(arr), _nd())
        update_halo!(arr3d; dims=_decomp_dims())
    else
        update_halo!(arr3d; dims=_decomp_dims(),
                     active_global_grid=_coarse_grid(size(arr)))
    end
end

# ── Direct MPI halo exchange (Dual / Bool eltypes) ───────────────────────────
# IGG's `update_halo!` rejects non-primitive eltypes (its `GGNumber` union),
# so `ForwardDiff.Dual` and `Bool` arrays take this manual `Sendrecv!` shift:
# 1-cell slabs per active dim, pre-allocated per-(eltype, shape, dim) buffers.

const _mpi_bufs = Dict{Tuple, NTuple{4,Array}}()

@inline function _get_mpi_bufs(::Type{T}, slab_shape::Tuple, dim::Int) where T
    key  = (T, slab_shape, dim)
    bufs = get(_mpi_bufs, key, nothing)   # single lookup; `get!` do-block allocates
    bufs === nothing || return bufs::NTuple{4,Array{T}}
    _mpi_bufs[key] = ntuple(_ -> zeros(T, slab_shape), 4)
end

function _slab(arr::AbstractArray, dim::Int, r::UnitRange)
    colons = ntuple(i -> i == dim ? r : (:), ndims(arr))
    @view arr[colons...]
end

function _scalar_halo_mpi!(arr::AbstractArray{T}) where T
    g    = _grid()
    N    = size(arr)
    comm = _comm()
    for dim in _decomp_dims()
        nleft  = g.neighbors[1, dim]
        nright = g.neighbors[2, dim]
        (nleft < 0 && nright < 0) && continue

        slab_shape = ntuple(i -> i == dim ? 1 : N[i], ndims(arr))
        send_left, recv_left, send_right, recv_right = _get_mpi_bufs(T, slab_shape, dim)

        # pack first/last interior cells (index 2 and N-1)
        copyto!(send_left,  _slab(arr, dim, 2:2))
        copyto!(send_right, _slab(arr, dim, N[dim]-1:N[dim]-1))

        # Classical shift pattern: send right / recv left, then the reverse.
        # Missing neighbors are already `MPI.PROC_NULL` (no-op slots), and the
        # pairing handles 2-rank periodic self-wrap without deadlock. For Dual,
        # `_wire_view` aliases buffers to flat V storage so MPI sees a native type.
        sR, rL = _wire_view(send_right), _wire_view(recv_left)
        sL, rR = _wire_view(send_left),  _wire_view(recv_right)
        MPI.Sendrecv!(sR, nright, dim*10,    rL, nleft,  dim*10,   comm, nothing)
        MPI.Sendrecv!(sL, nleft,  dim*10+1,  rR, nright, dim*10+1, comm, nothing)

        # unpack into ghost cells (index 1 and N)
        nleft  >= 0 && copyto!(_slab(arr, dim, 1:1),            recv_left)
        nright >= 0 && copyto!(_slab(arr, dim, N[dim]:N[dim]),  recv_right)
    end
end

# ── Unified scalar halo exchange ─────────────────────────────────────────────

@inline _wire_view(arr::Array) = arr
@inline _wire_view(arr::Array{<:Dual}) = _dual_view(arr)

@inline _native_eltype(::Type{<:Dual}) = false
@inline _native_eltype(::Type{Bool})   = false
@inline _native_eltype(::Type)         = true

function _do_scalar_halo!(arr::AbstractArray)
    _any_decomposed() || return
    _native_eltype(eltype(arr)) ? _scalar_halo_igg!(arr) : _scalar_halo_mpi!(arr)
end

# ── Vector (velocity-shaped) halo exchange ────────────────────────────────────

const _halo_bufs = Dict{Tuple, Array}()

_get_halo_buf(::Type{T}, dims::NTuple{N,Int}) where {T,N} = get!(() -> Array{T}(undef, dims), _halo_bufs, (T, dims))

# All components in ONE `update_halo!` call, on zero-copy 3D aliases of the
# contiguous component blocks: amortises IGG's per-call floor across the
# components and skips the scratch-buffer round-trip copies entirely.
# Coarse multigrid `L` arrays also land here (via `BC!` on `restrictL!`
# output), so component size routes to the per-size side-table like scalars.
function _velocity_halo_igg!(u::Array{T,N}) where {T,N}
    csize = size(u)[1:N-1]
    sz3   = ntuple(d -> d <= N-1 ? csize[d] : 1, 3)
    len   = prod(csize)
    comps = [unsafe_wrap(Array, pointer(u, (c-1)*len + 1), sz3) for c in 1:size(u, N)]
    if _is_fine_size(csize, _nd())
        GC.@preserve u update_halo!(comps...; dims=_decomp_dims())
    else
        GC.@preserve u update_halo!(comps...; dims=_decomp_dims(),
                                    active_global_grid=_coarse_grid(csize))
    end
end

function _do_velocity_halo!(u::AbstractArray{T,N}) where {T,N}
    _any_decomposed() || return
    if u isa Array && _native_eltype(T)
        _velocity_halo_igg!(u)
    else  # Dual / Bool / non-contiguous: per-component scratch-buffer path
        tmp = _get_halo_buf(T, size(u)[1:N-1])
        for d in 1:size(u, N)
            copyto!(tmp, @view u[.., d])
            _do_scalar_halo!(tmp)
            copyto!(@view(u[.., d]), tmp)
        end
    end
end

# ── ForwardDiff.Dual support ─────────────────────────────────────────────────
# `Dual{T,V,N}` is `isbits` with layout [value, partial₁, …, partialₙ] of `V`.
# Reinterpret to flat `V` for the wire, reconstruct on receive. SUM is
# element-wise on the flat view (derivative is linear); MIN/MAX are not —
# the partials of the extremum aren't the extrema of the partials.

@inline _dual_flat(x::Dual{T,V,N}) where {T,V,N} = V[value(x); partials(x).values...]
@inline _dual_unflat(::Type{Dual{T,V,N}}, f::AbstractVector{V}) where {T,V,N} =
    Dual{T,V,N}(f[1], Partials{N,V}(NTuple{N,V}(@view f[2:end])))

# Zero-copy flat-V alias of a contiguous Dual array. Callers must keep the
# parent rooted (GC.@preserve) for the lifetime of the view.
@inline function _dual_view(arr::Array{D}) where {Tg,V,N,D<:Dual{Tg,V,N}}
    unsafe_wrap(Array, Ptr{V}(pointer(arr)), length(arr) * (N + 1))
end

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

function _dual_arr_sum(x::AbstractArray{<:Dual})
    arr = x isa Array ? x : Array(x)
    GC.@preserve arr MPI.Allreduce!(_dual_view(arr), MPI.SUM, _comm())
    return arr
end

# ── Dispatch hooks for Parallel ──────────────────────────────────────────────
WaterLily._global_allreduce(x, ::Parallel) = MPI.Allreduce(x, MPI.SUM, _comm())
WaterLily._global_allreduce(x::Dual, ::Parallel) = _dual_sum(x)
WaterLily._global_allreduce(x::AbstractArray{<:Dual}, ::Parallel) = _dual_arr_sum(x)
WaterLily._global_min(a, b, ::Parallel) = MPI.Allreduce(min(a, b), MPI.MIN, _comm())
WaterLily._global_min(a::Dual, b::Dual, ::Parallel) = _dual_extremum(min(a, b), MPI.MIN)
WaterLily._global_max(x, ::Parallel) = MPI.Allreduce(x, MPI.MAX, _comm())
WaterLily._global_max(x::Dual, ::Parallel) = _dual_extremum(x, MPI.MAX)
WaterLily._scalar_halo!(x, ::Parallel) = _do_scalar_halo!(x)
WaterLily._velocity_halo!(u, ::Parallel) = _do_velocity_halo!(u)

# Decomposed periodic dims wrap through the halo (IGG periodic topology);
# non-decomposed ones wrap locally via `perBC!`, exactly as in serial.
# `perBC!` runs first so the halo slabs carry the wrapped corner values.
function WaterLily._comm!(a, perdir, ::Parallel)
    WaterLily.perBC!(a, _local_perdir(perdir))
    _do_scalar_halo!(a)
end
function WaterLily._velocity_comm!(a, perdir, ::Parallel)
    lp = _local_perdir(perdir)
    if !isempty(lp)
        for i in 1:size(a, ndims(a))
            WaterLily.perBC!(@view(a[.., i]), lp)
        end
    end
    _do_velocity_halo!(a)
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

# MG depth: distributed levels stop when the coarsest per-rank block is
# ~4–8 cells/dim — below that the ~20 μs/call IGG halo floor dominates the
# level, and the gathered rank-0 problem is still tiny. Global convergence
# past that point is owned by the agglomerated coarsest solve
# (`_coarsest_smooth!` below), so unlike the earlier global-size formula
# this needs no convergence safety margin. At np=1 (even with periodic
# self-wrap neighbors) keep serial full depth.
function WaterLily._mg_maxlevels(_dims, ::Parallel)
    _any_decomposed() || return 10
    g = _grid()
    max(2, floor(Int, log2(minimum(g.nxyz[d] - 2 for d in 1:_nd()) / 4)))
end

# ── Agglomerated coarsest-level solve ────────────────────────────────────────
# The distributed partition cannot coarsen past the per-rank block, so the
# coarsest reachable global grid GROWS with the rank count and domain-scale
# error needs ever more V-cycles (weak scaling: mean_iters 1.5 → 2.5 from
# np=8 to 512). Fix: gather the (tiny, ≤ a few MB) coarsest-level problem to
# rank 0 and continue the hierarchy there with the SERIAL multigrid, then
# broadcast the correction. `par_mode[]` is switched to `Serial()` on rank 0
# for the local solve — the hooks must not fire collectively while the other
# ranks wait at the broadcast.

const _agg_cache = Dict{Tuple, Any}()

# Run `f` with `par_mode[]` switched to `Serial` — rank-local solves must not
# fire collective hooks while the other ranks wait at the broadcast.
function _serially(f)
    old = WaterLily.par_mode[]
    WaterLily.par_mode[] = WaterLily.Serial()
    try f() finally WaterLily.par_mode[] = old end
end

function _agg_setup(p::WaterLily.Poisson{T}, m::Parallel) where T
    N, g    = ndims(p.r), _grid_for(size(p.r))
    loc_int = size(p.r) .- 2
    gl_int  = loc_int .* ntuple(d -> Int(g.dims[d]), N)
    loc     = Array{T}(undef, loc_int)                         # interior staging
    gint    = m.rank == 0 ? Array{T}(undef, gl_int) : nothing  # gather target
    x_g     = zeros(T, gl_int .+ 2)                            # broadcast buffer
    mlp     = m.rank != 0 ? nothing : _serially() do
        L_g, z_g = zeros(T, (gl_int .+ 2)..., N), zeros(T, gl_int .+ 2)
        try
            MultiLevelPoisson(x_g, L_g, z_g; perdir=p.perdir)
        catch  # gathered grid too small/odd for multigrid — plain Poisson
            WaterLily.Poisson(x_g, L_g, z_g; perdir=p.perdir)
        end
    end
    ofs   = ntuple(d -> Int(g.coords[d]) * loc_int[d], N)      # rank block in x_g
    block = CartesianIndices(ntuple(d -> ofs[d]+2:ofs[d]+1+loc_int[d], N))
    (; loc, gint, x_g, mlp, block, ins=WaterLily.inside(p.r))
end

# Gather the operator L to rank 0 and refresh the serial hierarchy there.
# Called from `update!(ml)` and the `MultiLevelPoisson` constructor — once
# per measurement, NOT per V-cycle (L is static between `measure!` calls).
function WaterLily._coarsest_update!(p::WaterLily.Poisson{T}, m::Parallel) where T
    _any_decomposed() && T <: Union{Float32, Float64} || return # Dual/AD: local path
    c, N = get!(() -> _agg_setup(p, m), _agg_cache, (T, size(p.r))), ndims(p.r)
    gin = WaterLily.inside(c.x_g)
    for d in 1:N
        copyto!(c.loc, @view p.L[c.ins, d])
        ImplicitGlobalGrid.gather!(c.loc, c.gint, m.comm)
        m.rank == 0 && (c.mlp.L[gin, d] .= c.gint)
    end
    if m.rank == 0
        _serially() do
            # periodic ghost wrap of the gathered L (zero is correct at walls only)
            WaterLily.BC!(c.mlp.L, zero(SVector{N,T}), false, p.perdir)
            WaterLily.update!(c.mlp)
        end
    end
end

function WaterLily._coarsest_smooth!(p::WaterLily.Poisson{T}, m::Parallel, ω) where T
    _any_decomposed() && T <: Union{Float32, Float64} ||       # Dual/AD: local path
        return WaterLily.smooth!(p; ω)
    c = get!(() -> _agg_setup(p, m), _agg_cache, (T, size(p.r)))
    copyto!(c.loc, @view p.r[c.ins])  # gather residual → rank-0 source
    ImplicitGlobalGrid.gather!(c.loc, c.gint, m.comm)
    if m.rank == 0  # serial solve of the global coarse problem: A ϵ = r
        c.mlp.z[WaterLily.inside(c.x_g)] .= c.gint
        fill!(c.mlp.x, zero(T)); empty!(c.mlp.n)
        _serially(() -> WaterLily.solver!(c.mlp; itmx=8))
    end
    MPI.Bcast!(c.x_g, 0, m.comm)
    @views p.ϵ[c.ins] .= c.x_g[c.block]
    WaterLily.increment!(p; ω)
end

end # module WaterLilyMPIExt

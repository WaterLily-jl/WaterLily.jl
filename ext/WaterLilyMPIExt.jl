# `Parallel` methods of WaterLily's `par_mode[]` hooks, and the native halo exchange
module WaterLilyMPIExt

using WaterLily
import WaterLily: @loop, _halo!, _sync
using MPI
using StaticArrays
using ForwardDiff: Dual, Partials, value, partials

struct Parallel <: WaterLily.AbstractParMode
    comm::MPI.Comm           # Cartesian communicator of the ranks
    rank::Int
    dims::Vector{Int}        # ranks per dim
    coords::Vector{Int}      # of this rank, from 0
    periods::Vector{Bool}
    neighbors::Matrix{Int}   # left and right rank per dim, negative (MPI.PROC_NULL) past a physical boundary
    nlocal::Vector{Int}      # interior cells of this rank per dim
    halo::WaterLily.HaloBackend
end

_par() = WaterLily.par_mode[]::Parallel
_comm() = _par().comm

WaterLily._mpi_rank(p::Parallel)   = p.rank
WaterLily._mpi_comm(p::Parallel)   = p.comm
WaterLily._mpi_nprocs(p::Parallel) = MPI.Comm_size(p.comm)

_nd() = length(_par().dims)
_any_decomposed() = any(>(1), _par().dims)
# periodic dims owned in full by every rank wrap with `perBC!`, the decomposed ones through the halo
_local_perdir(perdir) = (p = _par(); filter(j -> p.dims[j] == 1, perdir))

WaterLily._global_offset(::Val{N}, ::Type{T}, p::Parallel) where {N,T} =
    SVector{N,T}(ntuple(d -> T(get(p.coords, d, 0) * get(p.nlocal, d, 0)), N))

# the offset keeps a float or Dual eltype (precision, AD tag), Float32 (`loc`'s default) otherwise
WaterLily._loop_offset(::Type{T}, p::Parallel) where T<:Union{AbstractFloat,Dual} =
    WaterLily._global_offset(Val(_nd()), T, p)
WaterLily._loop_offset(::Type, p::Parallel) =
    WaterLily._global_offset(Val(_nd()), Float32, p)

# a body's `bbox` is in global cell indices: shift the block there and the result back, clipped to the block
function WaterLily._local_bbox(body, R::CartesianIndices{N}, p::Parallel) where N
    o = CartesianIndex(Tuple(WaterLily._global_offset(Val(N), Int, p)))
    (WaterLily.bbox(body, R .+ o) .- o) ∩ R
end

# the rank layout with the least halo surface per rank (`MPI.Dims_create`'s on ties)
function _shape_aware_topology(global_dims::NTuple{N,Int}, nprocs::Int) where N
    best_topo = Tuple(Int.(MPI.Dims_create(nprocs, zeros(Int, N))))
    best_score = if all(global_dims[d] % best_topo[d] == 0 for d in 1:N)
        _halo_surface(global_dims, best_topo)
    else
        typemax(Int)
    end
    function visit(remaining::Int, partial::Tuple)
        if length(partial) == N - 1
            topo = (partial..., remaining)
            all(global_dims[d] % topo[d] == 0 for d in 1:N) || return
            s = _halo_surface(global_dims, topo)
            if s < best_score
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

function WaterLily.init_waterlily_mpi(global_dims::NTuple{N}; perdir=(), select_device=!_gpu_bound(),
                                      halo=Symbol(get(ENV, "WATERLILY_HALO", "native"))) where N
    backend = get(WaterLily._halo_backends, Symbol(halo)) do
        error("no halo backend :$halo (have $(join(repr.(keys(WaterLily._halo_backends)), ", "))): " *
              "load ImplicitGlobalGrid for halo=:igg")
    end
    MPI.Initialized() || MPI.Init()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)

    mpi_dims   = _shape_aware_topology(global_dims, nprocs)
    local_dims = global_dims .÷ mpi_dims
    all(global_dims .== local_dims .* mpi_dims) ||
        error("Global dims $global_dims not evenly divisible by MPI topology " *
              "$mpi_dims with $nprocs ranks")

    periods = [d in perdir for d in 1:N]
    comm = MPI.Cart_create(MPI.COMM_WORLD, collect(mpi_dims); periodic=periods, reorder=true)
    me = MPI.Comm_rank(comm)
    neighbors = [MPI.Cart_shift(comm, d-1, 1)[s] for s in 1:2, d in 1:N]
    WaterLily.par_mode[] = Parallel(comm, me, collect(mpi_dims), MPI.Cart_coords(comm), periods, neighbors, collect(local_dims), backend)
    if select_device
        node = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, me)
        foreach(f -> f(MPI.Comm_rank(node), MPI.Comm_size(node)), WaterLily._device_selectors)
    end
    empty!(_halos)
    WaterLily._init_halo!(backend, comm, mpi_dims, Tuple(periods), local_dims)
    _overlap_min[] = parse(Int, get(ENV, "WATERLILY_OVERLAP_MIN", backend isa WaterLily.NativeHalo ? string(2^20) : string(typemax(Int))))
    _aware_min[] = (n = parse(Int, get(ENV, "WATERLILY_GPU_AWARE_MPI", "0")); n == 0 ? typemax(Int) : n)

    if me == 0
        topo = join(string.(mpi_dims), "×")
        loc  = join(string.(local_dims), "×")
        glob = join(string.(global_dims), "×")
        @info "WaterLily MPI: $(nprocs) ranks, topology=$(topo), " *
              "local=$(loc), global=$(glob), halo=$(halo)"
    end

    return local_dims, me, comm
end

# the launcher made one GPU visible to each rank
_gpu_bound() = any(("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")) do k
    v = get(ENV, k, ""); !isempty(v) && !occursin(',', v)
end

# Native halo: one message per neighbour rank, diagonals included, packed and unpacked by one kernel each,
# with a plan cached per array type and size. GPU halos are staged through host memory, or passed device to
# device from WATERLILY_GPU_AWARE_MPI bytes on (`0`: never, `1`: always).

const _halos = Dict{Tuple, Any}()
const _overlap_min = Ref(2^20)         # fewest cells of a loop that `comm!(f, ...)` splits
const _aware_min   = Ref(typemax(Int))  # fewest bytes of a halo passed to MPI in device memory

# the rank in direction `d` (-1, 0 or 1 per dim), or -1 past a domain boundary
function _peer(d)
    p = _par(); c = p.coords .+ collect(d)
    all(k -> p.periods[k] || 0 <= c[k] < p.dims[k], eachindex(c)) ? MPI.Cart_rank(p.comm, c) : -1
end

# the cells of `a` sent toward (`send`) or received from direction `d`
_halo_cells(a, d, send) = CartesianIndices(ntuple(ndims(a)) do k
    n, dk = size(a, k), get(d, k, 0)
    dk < 0 ? (send ? (2:2) : (1:1)) : dk > 0 ? (send ? (n-1:n-1) : (n:n)) :
    WaterLily.decomposed(k) ? ((WaterLily.phys_left(k) ? 1 : 2):(WaterLily.phys_right(k) ? n : n-1)) : (1:n)
end)
_dirkey(d) = evalpoly(3, d .+ 1)

function _halo_plan(a::AbstractArray{T}) where T
    dirs = [d for d in Iterators.product(ntuple(k -> WaterLily.decomposed(k) ? (-1:1) : (0:0), _nd())...)
            if any(!=(0), d) && _peer(d) >= 0]
    sort!(dirs, by=d -> count(!=(0), d))
    peers = unique(_peer.(dirs))
    msgs = [filter(d -> _peer(d) == p, dirs) for p in peers]
    # a message holds its regions in the sender's order; S and R index `a` for each buffer element
    L = LinearIndices(a); index(cells) = copyto!(similar(a, Int, length(cells)), cells)
    S = index([L[I] for m in msgs for d in sort(m, by=_dirkey) for I in _halo_cells(a, d, true)])
    R = index([L[I] for m in msgs for d in sort(m, by=d -> _dirkey(.-d)) for I in _halo_cells(a, d, false)])
    lens = [sum(d -> length(_halo_cells(a, d, true)), m) for m in msgs]; ends = cumsum(lens)
    buffers(n) = n * sizeof(T) >= _aware_min[] ? (b = similar(a, T, n); (b, b)) : WaterLily._host_buffers(a, T, n)
    (s, sh), (r, rh) = buffers(ends[end]), buffers(ends[end])
    (; S, R, s, sh, r, rh, peers, ranges=UnitRange.(ends .- lens .+ 1, ends),
       sreq=MPI.UnsafeMultiRequest(length(peers)), rreq=MPI.UnsafeMultiRequest(length(peers)))
end

_halo!(work, a) = _any_decomposed() ? _halo!(work, a, _par().halo) : work()
_halo!(a) = _halo!(Returns(nothing), a)
WaterLily._init_halo!(::WaterLily.HaloBackend, comm, dims, periods, nlocal) = nothing

_halo!(work, a, ::WaterLily.NativeHalo) = _exchange!(work, a, get!(() -> _halo_plan(a), _halos, (typeof(a), size(a))))
function _exchange!(work, a, h)
    WaterLily.pack!(h.s, a, h.S)
    _sync(a)  # the pack is done, and the last unpack from h.r
    for (i, p) in enumerate(h.peers)
        MPI.Irecv!(view(h.rh, h.ranges[i]), _comm(), h.rreq[i]; source=p, tag=0)
    end
    for (i, p) in enumerate(h.peers)
        MPI.Isend(view(h.sh, h.ranges[i]), _comm(), h.sreq[i]; dest=p, tag=0)
    end
    work()
    MPI.Waitall(h.rreq)
    WaterLily.unpack!(a, h.r, h.R)  # not waited for: later kernels queue after it
    MPI.Waitall(h.sreq)
end

# `Dual`s go to MPI as their flat values and partials: a sum is element-wise, a min or max is the extremum's Dual

@inline _dual_flat(x::Dual{T,V,N}) where {T,V,N} = V[value(x); partials(x).values...]
@inline _dual_unflat(::Type{Dual{T,V,N}}, f::AbstractVector{V}) where {T,V,N} =
    Dual{T,V,N}(f[1], Partials{N,V}(NTuple{N,V}(@view f[2:end])))

# flat `V` alias of a Dual array (keep `arr` rooted with GC.@preserve; MPI.jl takes no host ReinterpretArray)
@inline function _dual_view(arr::Array{D}) where {Tg,V,N,D<:Dual{Tg,V,N}}
    unsafe_wrap(Array, Ptr{V}(pointer(arr)), length(arr) * (N + 1))
end
@inline _dual_view(arr::AbstractArray{<:Dual{Tg,V}}) where {Tg,V} = reinterpret(V, vec(arr))

_dual_sum(x::Dual{T,V,N}) where {T,V,N} =
    (f = _dual_flat(x); MPI.Allreduce!(f, MPI.SUM, _comm()); _dual_unflat(Dual{T,V,N}, f))

function _dual_extremum(x::Dual{T,V,N}, op) where {T,V,N}
    comm = _comm()
    vals = MPI.Allgather(value(x), comm)
    root = (op === MPI.MIN ? argmin(vals) : argmax(vals)) - 1
    f = MPI.Comm_rank(comm) == root ? _dual_flat(x) : zeros(V, N + 1)
    MPI.Bcast!(f, root, comm); _dual_unflat(Dual{T,V,N}, f)
end

WaterLily._global_allreduce(x, ::Parallel) = MPI.Allreduce(x, MPI.SUM, _comm())
WaterLily._global_allreduce(x::AbstractArray, p::Parallel) = WaterLily._global_allreduce!(copy(x), p)
WaterLily._global_allreduce(x::Dual, ::Parallel) = _dual_sum(x)
WaterLily._global_allreduce!(x::AbstractArray, ::Parallel) = (_sync(x); MPI.Allreduce!(x, MPI.SUM, _comm()); x)
WaterLily._global_allreduce!(x::AbstractArray{<:Dual}, ::Parallel) =
    (_sync(x); GC.@preserve x MPI.Allreduce!(_dual_view(x), MPI.SUM, _comm()); x)
WaterLily._global_min(a, b, ::Parallel) = MPI.Allreduce(min(a, b), MPI.MIN, _comm())
WaterLily._global_min(a::Dual, b::Dual, ::Parallel) = _dual_extremum(min(a, b), MPI.MIN)
WaterLily._global_max(x, ::Parallel) = MPI.Allreduce(x, MPI.MAX, _comm())
WaterLily._global_max(x::Dual, ::Parallel) = _dual_extremum(x, MPI.MAX)
# one Allgather (MPI.jl has no user-defined reductions on ARM), summed in rank order: the same bits on every rank
function WaterLily._global_sum_max(s::T, m::T, ::Parallel) where T<:AbstractFloat
    v = MPI.Allgather((s, m), _comm())
    (foldl((a, x) -> a + x[1], v; init=zero(T)), maximum(last, v))
end
WaterLily._global_sum_max(s, m, p::Parallel) = (WaterLily._global_allreduce(s, p), WaterLily._global_max(m, p))
WaterLily._scalar_halo!(x, ::Parallel) = _halo!(x)
WaterLily._velocity_halo!(u, ::Parallel) = _halo!(u)

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

# `f(R)`, with the halo of `a` travelling during `f(inner(R))` for long enough loops
function _overlap!(f, a, R)
    _any_decomposed() && length(R) >= _overlap_min[] || return (_halo!(a); f(R))
    _halo!(() -> f(WaterLily.inner(R)), a)
    f(WaterLily.shell(R))
end

WaterLily._phys_left(j,  p::Parallel) = p.neighbors[1, j] < 0
WaterLily._phys_right(j, p::Parallel) = p.neighbors[2, j] < 0

WaterLily._decomposed(j, p::Parallel) = j <= length(p.dims) && p.dims[j] > 1

end # module WaterLilyMPIExt

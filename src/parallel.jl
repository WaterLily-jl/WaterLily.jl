# Parallel hooks dispatch on `par_mode[]` (`Serial()` here; WaterLilyMPIExt adds `Parallel` methods)
using LinearAlgebra: ⋅
using StaticArrays

abstract type AbstractParMode end
struct Serial <: AbstractParMode end
const par_mode = Ref{AbstractParMode}(Serial())

local_dot(a, b) = a⋅b
local_sum(a) = sum(a)
local_sumabs(a) = sum(abs, a)

"""
    mpi_rank(), mpi_comm(), mpi_nprocs()

The rank, communicator and number of ranks (`0`, `nothing`, `1` in serial).
"""
mpi_rank()   = _mpi_rank(par_mode[])
mpi_comm()   = _mpi_comm(par_mode[])
mpi_nprocs() = _mpi_nprocs(par_mode[])
_mpi_rank(::Serial)   = 0
_mpi_comm(::Serial)   = nothing
_mpi_nprocs(::Serial) = 1

"""
    @distributed Simulation(dims, uBC, L; kwargs...)

Call `init_waterlily_mpi(dims; perdir)` (needs `using MPI`) and construct with the rank-local `dims`.
Works for any constructor taking the global `dims` first, with `perdir` written as a keyword of the call.
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
    # keywords after `;` are a leading :parameters block, those after `,` trailing :kw arguments
    has_params = length(ex.args) >= 2 && ex.args[2] isa Expr && ex.args[2].head === :parameters
    dims_idx   = has_params ? 3 : 2
    (length(ex.args) >= dims_idx && !_iskw(ex.args[dims_idx])) ||
        error("@distributed: `$(f)(...)` has no positional `dims` argument (it must come first)")
    global_dims = ex.args[dims_idx]

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

# The rank's offset that `@loop` adds to `loc(...)` in its body (`nothing` in serial)
_loop_offset(::Type{T}) where T = _loop_offset(T, par_mode[])
_loop_offset(::Type{T}, ::Serial) where T = nothing

"""
    global_dot(a, b), global_sum(a), global_length(r), global_min(a, b), global_max(x)

Reductions across MPI ranks (the local value in serial).
"""
global_dot(a, b) = global_allreduce(local_dot(a, b))
global_sum(a) = global_allreduce(local_sum(a))
global_length(r) = global_allreduce(length(r))
global_min(a, b) = _global_min(a, b, par_mode[])
_global_min(a, b, ::Serial) = min(a, b)
global_max(x) = _global_max(x, par_mode[])
_global_max(x, ::Serial) = x
# `(global_allreduce(s), global_max(m))` in one reduction
global_sum_max(s, m) = _global_sum_max(s, m, par_mode[])
_global_sum_max(s, m, ::Serial) = (s, m)

"""
    global_allreduce(x), global_allreduce!(x)

Sum of `x` (a scalar or an array, element-wise) across MPI ranks; `global_allreduce!` in place.
"""
global_allreduce(x) = _global_allreduce(x, par_mode[])
_global_allreduce(x, ::Serial) = x
global_allreduce!(x) = _global_allreduce!(x, par_mode[])
_global_allreduce!(x, ::Serial) = x

"""
    scalar_halo!(x), velocity_halo!(u)

Fill the ghosts of `x`, or of all components of `u` (every dim past the spatial ones), from the neighbour ranks.
No-op in serial.
"""
scalar_halo!(x) = _scalar_halo!(x, par_mode[])
velocity_halo!(u) = _velocity_halo!(u, par_mode[])
_scalar_halo!(x, ::Serial) = nothing
_velocity_halo!(u, ::Serial) = nothing
# Halo backends, picked by `init_waterlily_mpi(...; halo)` from `_halo_backends`: `_halo!(work, a, backend)` fills
# the ghosts of `a` and runs `work()`; `_init_halo!(backend, comm, dims, periods, nlocal)` sets it up
abstract type HaloBackend end
struct NativeHalo <: HaloBackend end
function _halo! end
function _init_halo! end
const _halo_backends = Dict{Symbol,HaloBackend}(:native => NativeHalo())
# wait for the kernels writing `a` before MPI reads it
_sync(a) = KernelAbstractions.synchronize(KernelAbstractions.get_backend(a))
# turns the GPU package's implicit stream synchronization off for a halo array (WaterLilyCUDAExt)
_no_stream_sync!(a) = a
# a halo buffer of `n` elements for kernels on `a`'s backend, and its host view for MPI
_host_buffers(a, T, n) = (b = similar(a, T, n); (b, b))
# GPU selection of each GPU extension loaded, called with node-local rank `l` of `n`
const _device_selectors = Function[]
_node_gpu(l, n, ndev) = n <= ndev ? l : error("more MPI ranks on a node ($n) than GPUs ($ndev): start one rank per GPU, " *
    "make one GPU visible to each rank (CUDA_VISIBLE_DEVICES, ROCR_VISIBLE_DEVICES), or pass select_device=false")

# whether the left/right side of direction `j` is a domain boundary, not a neighbour rank
phys_left(j)  = _phys_left(j, par_mode[])
phys_right(j) = _phys_right(j, par_mode[])
_phys_left(j, ::Serial)  = true
_phys_right(j, ::Serial) = true

"""
    comm!(a, perdir), comm!(f, a, perdir, R=inside(a))

`perBC!` and the halo of `a`. With `f`, then `f(R)`: `f` loops a 7-point stencil of `a`
over the range it gets, `inner(R)` while the halo messages travel and `shell(R)` after.
"""
comm!(a, perdir) = _comm!(a, perdir, par_mode[])
_comm!(a, perdir, ::Serial) = perBC!(a, perdir)
comm!(f::F, a, perdir, R=inside(a)) where F = _comm!(f, a, perdir, R, par_mode[])
_comm!(f, a, perdir, R, ::Serial) = (perBC!(a, perdir); f(R))

# the cells of `R` whose 7-point stencil reads no ghost of a decomposed dim, and the others (as one range)
inner(R::CartesianIndices{N}) where N =
    CartesianIndices(ntuple(j -> decomposed(j) ? (first(R.indices[j])+1:last(R.indices[j])-1) : R.indices[j], N))
function shell(R::CartesianIndices{N}) where N
    r, i = R.indices, inner(R).indices
    layer(j, k) = CartesianIndices(ntuple(d -> d < j ? i[d] : d == j ? k : r[d], N))
    chain(ntuple(2N) do m
        j = cld(m, 2); a, b = first(r[j]), last(r[j])
        !decomposed(j) ? layer(j, a:a-1) : isodd(m) ? layer(j, a:a) : layer(j, max(a+1, b):b)
    end)
end

"""
    velocity_comm!(a, perdir), velocity_comm!(f, a, perdir, R)

`comm!` for every component of the velocity `a`.
"""
velocity_comm!(a, perdir) = _velocity_comm!(a, perdir, par_mode[])
velocity_comm!(f::F, a, perdir, R) where F = _velocity_comm!(f, a, perdir, R, par_mode[])
_velocity_comm!(f, a, perdir, R, ::Serial) = (_velocity_comm!(a, perdir, Serial()); f(R))
function _velocity_comm!(a, perdir, ::Serial)
    _, n = size_u(a)
    for i ∈ 1:n
        perBC!(selectdim(a, ndims(a), i), perdir)
    end
end

# whether direction `j` is split across ranks (a split periodic one takes the Neumann fluxes in `conv_diff!`)
decomposed(j) = _decomposed(j, par_mode[])
_decomposed(j, ::Serial) = false

"""
    global_offset(Val(N), T=Float32)

The global coordinates of this rank's origin (zero in serial).
"""
global_offset(::Val{N}, ::Type{T}=Float32) where {N,T} = _global_offset(Val(N), T, par_mode[])
global_offset(N::Int, T::Type=Float32) = global_offset(Val(N), T)
_global_offset(::Val{N}, ::Type{T}, ::Serial) where {N,T} = zero(SVector{N,T})

"""
    init_waterlily_mpi(global_dims; perdir=(), select_device, halo) → (local_dims, rank, comm)

Split `global_dims` over the MPI ranks (needs `using MPI`). `select_device` gives each node-local rank its own GPU
(by default unless the launcher binds one GPU per rank); `halo` picks the halo backend, `:native` or `:igg`
(with ImplicitGlobalGrid; default `ENV["WATERLILY_HALO"]`).
"""
function init_waterlily_mpi end

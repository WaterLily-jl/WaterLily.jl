"""
WaterLilyIGGExt — activated when ImplicitGlobalGrid is loaded with WaterLily and
MPI. Registers the halo backend `:igg`: with `init_waterlily_mpi(...; halo=:igg)`
(or `WATERLILY_HALO=igg`), ImplicitGlobalGrid's `update_halo!` exchanges the
halos of `Float32` and `Float64` arrays, one decomposed dim after the other, on
global grids of the rank layout WaterLilyMPIExt picked. Other eltypes (`Dual`,
`Bool`, which IGG rejects) take the native exchange, and the work of
`comm!(f, ...)` runs after the halo.
"""
module WaterLilyIGGExt

using WaterLily, MPI, ImplicitGlobalGrid
import WaterLily: _halo!, _init_halo!, _sync, _no_stream_sync!
const IGG = ImplicitGlobalGrid

struct IGGHalo <: WaterLily.HaloBackend end
__init__() = (WaterLily._halo_backends[:igg] = IGGHalo())

# One global grid per rank-local array size (fine and coarse levels), built lazily
# from the same keywords: the Cartesian communicator of WaterLilyMPIExt, whose
# ranks and coordinates `reorder=0` keeps, its ranks per dim and periods (IGG
# always takes three dims). `update_halo!` covers the decomposed dims only:
# periodic dims owned in full by every rank wrap through `comm!`'s `perBC!`.
const _grids = Dict{NTuple{3,Int},IGG.GlobalGrid}()
const _kw = Ref{NamedTuple}()
const _dims = Ref{Tuple}()   # the decomposed ones
const _nd = Ref(0)           # spatial dims
_pad(x, v) = ntuple(d -> get(x, d, v), 3)

function _init_halo!(::IGGHalo, comm, dims, periods, nlocal)
    isassigned(_kw) && finalize_global_grid(finalize_MPI=false)  # re-init in the same session
    init_global_grid(; init_MPI=false, select_device=false)       # IGG's package init, no grid
    empty!(_grids)
    (dimx, dimy, dimz), (periodx, periody, periodz) = _pad(dims, 1), _pad(periods, false)
    _kw[] = (; dimx, dimy, dimz, periodx, periody, periodz, overlaps=(2, 2, 2), halowidths=(1, 1, 1),
             reorder=0, comm, quiet=true)
    _dims[], _nd[] = Tuple(d for d in eachindex(dims) if dims[d] > 1), length(dims)
end
_grid(sz) = get!(() -> create_global_grid(sz...; _kw[]...), _grids, sz)

# A scalar array as itself, a velocity-shaped array as its components, each as
# a 3D array on the same memory: one `update_halo!` call for all of them
_fields(a::Array, sz, nc) = [unsafe_wrap(Array, pointer(a, (c-1)*prod(sz) + 1), sz) for c in 1:nc]
_fields(a::DenseArray, sz, nc) = [reshape(selectdim(reshape(a, sz..., nc), 4, c), sz) for c in 1:nc]

function _halo!(work, a::DenseArray{<:Union{Float32,Float64}}, ::IGGHalo)
    sz = _pad(size(a)[1:_nd[]], 1)  # a velocity-shaped array has its components along the last dim
    _sync(a); _no_stream_sync!(a)    # IGG packs and unpacks on streams of its own
    GC.@preserve a update_halo!(_fields(a, sz, length(a) ÷ prod(sz))...; dims=_dims[], active_global_grid=_grid(sz))
    work()
end
_halo!(work, a, ::IGGHalo) = _halo!(work, a, WaterLily.NativeHalo())

end # module WaterLilyIGGExt

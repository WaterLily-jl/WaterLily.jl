# Halo backend `:igg`: ImplicitGlobalGrid's `update_halo!` for Float32/Float64 arrays, the native exchange otherwise
module WaterLilyIGGExt

using WaterLily, MPI, ImplicitGlobalGrid
import WaterLily: _halo!, _init_halo!, _sync, _no_stream_sync!
const IGG = ImplicitGlobalGrid

struct IGGHalo <: WaterLily.HaloBackend end
__init__() = (WaterLily._halo_backends[:igg] = IGGHalo())

# one IGG grid per array size, on WaterLilyMPIExt's Cartesian communicator (`reorder=0` keeps its ranks)
const _grids = Dict{NTuple{3,Int},IGG.GlobalGrid}()
const _kw = Ref{NamedTuple}()
const _dims = Ref{Tuple}()   # the decomposed ones
const _nd = Ref(0)           # spatial dims
_pad(x, v) = ntuple(d -> get(x, d, v), 3)

function _init_halo!(::IGGHalo, comm, dims, periods, nlocal)
    isassigned(_kw) && finalize_global_grid(finalize_MPI=false)
    init_global_grid(; init_MPI=false, select_device=false)
    empty!(_grids)
    (dimx, dimy, dimz), (periodx, periody, periodz) = _pad(dims, 1), _pad(periods, false)
    _kw[] = (; dimx, dimy, dimz, periodx, periody, periodz, overlaps=(2, 2, 2), halowidths=(1, 1, 1),
             reorder=0, comm, quiet=true)
    _dims[], _nd[] = Tuple(d for d in eachindex(dims) if dims[d] > 1), length(dims)
end
_grid(sz) = get!(() -> create_global_grid(sz...; _kw[]...), _grids, sz)

# the components of `a` as 3D arrays on its memory, for one `update_halo!` call
_fields(a::Array, sz, nc) = [unsafe_wrap(Array, pointer(a, (c-1)*prod(sz) + 1), sz) for c in 1:nc]
_fields(a::DenseArray, sz, nc) = [reshape(selectdim(reshape(a, sz..., nc), 4, c), sz) for c in 1:nc]

function _halo!(work, a::DenseArray{<:Union{Float32,Float64}}, ::IGGHalo)
    sz = _pad(size(a)[1:_nd[]], 1)
    _sync(a); _no_stream_sync!(a)    # IGG packs and unpacks on streams of its own
    GC.@preserve a update_halo!(_fields(a, sz, length(a) ÷ prod(sz))...; dims=_dims[], active_global_grid=_grid(sz))
    work()
end
_halo!(work, a, ::IGGHalo) = _halo!(work, a, WaterLily.NativeHalo())

end # module WaterLilyIGGExt

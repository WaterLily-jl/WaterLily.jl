module WaterLilyWriteVTKExt

using WriteVTK, WaterLily
import WaterLily: vtkWriter, save!, default_attrib, pvd_collection, vtk_attribs
using Printf: @sprintf
import Base: close

"""
    pvd_collection(fname; append=false)

Wrapper for `paraview_collection` returning a pvd file header.
"""
pvd_collection(fname;append=false) = paraview_collection(fname;append=append)
"""
    VTKWriter(fname; attrib, dir, T, body_mask=false, include_bc=false)

Holds the collection name and `Dict{String,Function}` attributes for VTK output.
Defaults save `Velocity` and `Pressure`. With `include_bc=true`, the `buff=1`
ghost layer is retained along non-periodic physical boundaries (under MPI, only
by the rank that owns the global boundary in that direction).
"""
struct VTKWriter
    fname         :: String
    dir_name      :: String
    collection    :: WriteVTK.CollectionFile
    output_attrib :: Dict{String,Function}
    count         :: Vector{Int}
    body_mask     :: Bool
    include_bc    :: Bool
end
function vtkWriter(fname="WaterLily";attrib=default_attrib(),dir="vtk_data",T=Float32,body_mask=false,include_bc=false)
    mkpath(dir)  # race-safe: multiple MPI ranks may enter here at once
    VTKWriter(fname,dir,pvd_collection(fname),attrib,[0],body_mask,include_bc)
end
function vtkWriter(fname,dir::String,collection,attrib::Dict{String,Function},k,body_mask::Bool=false,include_bc::Bool=false)
    VTKWriter(fname,dir,collection,attrib,[k],body_mask,include_bc)
end
"""
    default_attrib()

Returns a `Dict{String,Function}` of attribute name → simulation accessor; the
function builds the data written under that VTK key. Extend by passing your
own dict to `vtkWriter`.
"""
_velocity(a::AbstractSimulation) = a.flow.u
_pressure(a::AbstractSimulation) = a.flow.p
default_attrib() = Dict("Velocity"=>_velocity, "Pressure"=>_pressure)

# Optional attribute accessors. None added to `default_attrib()` — opt in via
#     vtkWriter("name"; attrib = merge(default_attrib(), vtk_attribs(:sdf, :rank)))
# All write into `flow.σ` (scratch) and return it; `_interior` then copies before
# the next call clobbers σ, so multiple selections compose cleanly.
function _sdf_attrib(a::AbstractSimulation)
    σ, body, t, T = a.flow.σ, a.body, sum(a.flow.Δt), eltype(a.flow.σ)
    # Interior via measure_sdf! preserves the flood-fill sign flip for closed MeshBodies.
    WaterLily.measure_sdf!(σ, body, t; fastd²=T(Inf))
    # Fill the ghost layer too so `include_bc=true` displays real geometry rather than
    # stale/zero σ. The natural BVH sign is correct for cells outside the body, which
    # is the usual case for ghosts; the flood-fill flip is interior-only.
    Rin = inside(σ)
    WaterLily.@loop σ[I] = (I in Rin ? σ[I] : WaterLily.sdf(body, loc(0,I,T), t)) over I ∈ CartesianIndices(σ)
    return σ
end
function _vorticity_attrib(a::AbstractSimulation)
    fill!(a.flow.σ, zero(eltype(a.flow.σ)))   # clear ghosts so they don't carry the previous attrib's σ
    if ndims(a.flow.p) == 2
        WaterLily.@inside a.flow.σ[I] = WaterLily.curl(3, I, a.flow.u)
    else
        WaterLily.@inside a.flow.σ[I] = WaterLily.ω_mag(I, a.flow.u)
    end
    return a.flow.σ
end
function _lambda2_attrib(a::AbstractSimulation)
    @assert ndims(a.flow.p) == 3 "λ₂ is 3D-only"
    fill!(a.flow.σ, zero(eltype(a.flow.σ)))
    WaterLily.@inside a.flow.σ[I] = WaterLily.λ₂(I, a.flow.u)
    return a.flow.σ
end
_rank_attrib(a::AbstractSimulation) = (a.flow.σ .= WaterLily.mpi_rank(); a.flow.σ)

const _attrib_lib = Dict(
    :sdf       => ("SDF",       _sdf_attrib),
    :vorticity => ("Vorticity", _vorticity_attrib),
    :lambda2   => ("Lambda2",   _lambda2_attrib),
    :rank      => ("Rank",      _rank_attrib),
)

"""
    vtk_attribs(syms::Symbol...) → Dict{String,Function}

Pre-built attribute accessors for `vtkWriter`. Available keys: `:sdf`,
`:vorticity` (ω₃ in 2D, |ω| in 3D), `:lambda2` (3D only), `:rank`. Merge into
`default_attrib()` to opt in.
"""
function vtk_attribs(syms::Symbol...)
    d = Dict{String,Function}()
    for s in syms
        haskey(_attrib_lib, s) || error("unknown vtk attrib :$s. Available: $(collect(keys(_attrib_lib)))")
        name, func = _attrib_lib[s]; d[name] = func
    end
    return d
end

# Strip the `buff=1` ghost layer along the first `nd` spatial dims. `lo[d]`/`hi[d]` ∈ {0,1}
# keep one BC cell on that side. Returns a copy when slicing, the original array otherwise.
function _interior(data::AbstractArray, nd::Int, N_int::NTuple;
                   lo::NTuple=ntuple(_->0, nd), hi::NTuple=ntuple(_->0, nd))
    size(data)[1:nd] == N_int && return data
    axs = ntuple(d -> d <= nd ? ((2-lo[d]):(size(data,d)-1+hi[d])) : Colon(), ndims(data))
    return Array(@view data[axs...])
end

# Cell-centred BDIM mask `μ₀(sdf, ϵ) ∈ [0,1]`. Reuses `flow.σ` (scratch between mom_steps)
# and uses `fastd²=(ϵ+1)²` so the BVH closest-point search short-circuits far from the body
# — cells beyond saturate to μ₀=1 anyway. Ghost cells are filled too; otherwise stale σ at
# the BC layer would yield mask=μ₀(0,ϵ)=0.5 and scale every output field down. Returns an
# owned buffer (`_interior` copies).
function _body_mask(a::AbstractSimulation, nd::Int, N_int::NTuple; kw...)
    σ, body, t, T = a.flow.σ, a.body, sum(a.flow.Δt), eltype(a.flow.σ)
    WaterLily.measure_sdf!(σ, body, t; fastd²=T((a.ϵ+1)^2))
    Rin = inside(σ)
    WaterLily.@loop σ[I] = (I in Rin ? σ[I] : WaterLily.sdf(body, loc(0,I,T), t)) over I ∈ CartesianIndices(σ)
    σ_int = _interior(σ, nd, N_int; kw...)
    σ_int .= WaterLily.μ₀.(σ_int, T(a.ϵ))
    return σ_int
end

# Multiply `data` by `mask` in place. Scalar fields share the mask shape; vector fields
# carry an extra trailing axis broadcast across components. `data` must be owned.
function _apply_mask!(data::AbstractArray, mask::AbstractArray, nd::Int)
    if ndims(data) == nd
        data .*= mask
    else
        for c in 1:size(data, ndims(data))
            selectdim(data, ndims(data), c) .*= mask
        end
    end
    return data
end

# Common attribute-write loop shared by serial and MPI paths.
function _write_attribs!(vtk, w::VTKWriter, a::AbstractSimulation, nd::Int, N_int::NTuple, lo::NTuple, hi::NTuple)
    mask = w.body_mask ? _body_mask(a, nd, N_int; lo, hi) : nothing
    for (name, func) in w.output_attrib
        data = _interior(func(a), nd, N_int; lo, hi)
        isnothing(mask) || _apply_mask!(data, mask, nd)
        vtk[name, VTKCellData()] = ndims(data) > nd ? components_first(data) : data
    end
end

"""
    save!(w::VTKWriter, a::AbstractSimulation)

Write `a` to VTK. Dispatches on `par_mode[]`: serial → `.vti`; parallel → per-rank
`.vti` pieces with a `.pvti` header on rank 0. Ghost layers are stripped automatically.
"""
function save!(w::VTKWriter, a::AbstractSimulation)
    # Refresh velocity ghosts so `include_bc=true` shows BC values, not stale data.
    w.include_bc && WaterLily.BC!(a.flow.u, a.flow.uBC, a.flow.exitBC, a.flow.perdir, sum(a.flow.Δt))
    _save!(w, a, WaterLily.par_mode[])
    # Attribs (sdf/rank/body_mask) leak values into σ ghosts; CFL reads `maximum(a.σ)`
    # over the full array, so leave σ zeroed for the next step.
    fill!(a.flow.σ, zero(eltype(a.flow.σ)))
end

function _save!(w::VTKWriter, a::AbstractSimulation, ::WaterLily.Serial)
    k = w.count[1]; nd = ndims(a.flow.p); N_int = size(a.flow.p) .- 2
    pad = ntuple(d -> w.include_bc && !(d in a.flow.perdir) ? 1 : 0, nd)
    vtk = vtk_grid(w.dir_name*@sprintf("/%s_%06i", w.fname, k), [1:n+1 for n in N_int .+ 2 .* pad]...)
    _write_attribs!(vtk, w, a, nd, N_int, pad, pad)
    vtk_save(vtk); w.count[1] = k+1
    w.collection[round(sim_time(a),digits=4)] = vtk
end

function _save!(w::VTKWriter, a::AbstractSimulation, ::WaterLily.AbstractParMode)
    mpi_id = Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI")
    igg_id = Base.PkgId(Base.UUID("4d7a3746-15be-11ea-1130-334b0c4f5fa0"), "ImplicitGlobalGrid")
    (haskey(Base.loaded_modules, mpi_id) && haskey(Base.loaded_modules, igg_id)) ||
        return _save!(w, a, WaterLily.Serial())

    MPIMod, IGG = Base.loaded_modules[mpi_id], Base.loaded_modules[igg_id]
    comm = MPIMod.COMM_WORLD; me = MPIMod.Comm_rank(comm); nprocs = MPIMod.Comm_size(comm)
    nprocs == 1 && return _save!(w, a, WaterLily.Serial())

    g  = IGG.global_grid()
    nd = sum(g.nxyz .> 1)
    nx = ntuple(d -> g.nxyz[d] - g.overlaps[d], nd)
    coords_mat = reshape(MPIMod.Allgather(Int32[g.coords[d] for d in 1:nd], comm), nd, nprocs)
    dims_proc  = ntuple(d -> Int(maximum(@view coords_mat[d, :])) + 1, nd)
    bc = ntuple(d -> w.include_bc && !(d in a.flow.perdir) ? 1 : 0, nd)

    # Per-rank VTK extents: leftmost/rightmost rank owns the BC cell, middle ranks shift by bc.
    extents = [ntuple(d -> begin
                 c = Int(coords_mat[d, r])
                 lo = c == 0 ? bc[d] : 0; hi = c == dims_proc[d]-1 ? bc[d] : 0
                 s  = c*nx[d] + (c > 0)*bc[d]
                 (s+1):(s + nx[d] + lo + hi + 1)
               end, nd) for r in 1:nprocs]
    lo = ntuple(d -> Int(coords_mat[d,me+1])==0              ? bc[d] : 0, nd)
    hi = ntuple(d -> Int(coords_mat[d,me+1])==dims_proc[d]-1 ? bc[d] : 0, nd)

    k = w.count[1]
    fname = joinpath(w.dir_name, @sprintf("%s_%06i", w.fname, k))
    pvtk  = pvtk_grid(fname, extents[me+1]...; part=me+1, extents=extents)
    _write_attribs!(pvtk, w, a, nd, size(a.flow.p) .- 2, lo, hi)
    me == 0 ? (w.collection[round(sim_time(a), digits=4)] = pvtk) : vtk_save(pvtk)
    w.count[1] = k+1
end

"""
    close(w::VTKWriter)

Flush the pvd collection file. Must be called to finalise the dataset.
"""
close(w::VTKWriter) = (vtk_save(w.collection); nothing)

# Move the trailing component axis of a vector field to the front (VTK requirement).
components_first(a::AbstractArray{T,N}) where {T,N} = permutedims(a, [N, 1:N-1...])

end # module

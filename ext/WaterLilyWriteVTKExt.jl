module WaterLilyWriteVTKExt

using WriteVTK, WaterLily
import WaterLily: vtkWriter, save!, default_attrib, pvd_collection
using Printf: @sprintf
import Base: close

"""
    pvd_collection(fname;append=false)

Wrapper for a `paraview_collection` that returns a pvd file header
"""
pvd_collection(fname;append=false) = paraview_collection(fname;append=append)
"""
    VTKWriter(fname; attrib, dir, T, body_mask=false, include_bc=false)

Generates a `VTKWriter` that hold the collection name to which the `vtk` files are written.
The default attributes that are saved are the `Velocity` and the `Pressure` fields.
Custom attributes can be passed as `Dict{String,Function}` to the `attrib` keyword.

If `include_bc=true`, the boundary layer (index 1 / M with the layout's `buff=1`)
is kept along every non-periodic dimension. Useful for visually debugging BC
behaviour. Under MPI, the layer is kept only by the rank that owns the global
physical boundary in that dimension; MPI-internal seams stay at the interior
slice.
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

Returns a `Dict` containing the `name` and `bound_function` for the default attributes.
The `name` is used as the key in the `vtk` file and the `bound_function` generates the data
to put in the file. With this approach, any variable can be save to the vtk file.
"""
_velocity(a::AbstractSimulation) = a.flow.u
_pressure(a::AbstractSimulation) = a.flow.p
default_attrib() = Dict("Velocity"=>_velocity, "Pressure"=>_pressure)

"""
    _interior(data, nd, N_int; lo, hi)

Strip the `buff=1` ghost layer along each of the first `nd` spatial dims.
`lo[d]` / `hi[d]` ∈ {0,1} widen the slice by one to keep the BC layer on
that side; defaults to 0 (strip). If `data` is already at interior size,
pass through untouched (no widening possible).
"""
function _interior(data::AbstractArray, nd::Int, N_int::NTuple;
                   lo::NTuple=ntuple(_->0, nd), hi::NTuple=ntuple(_->0, nd))
    size(data)[1:nd] == N_int && return data
    axs = ntuple(d -> d <= nd ? ((2-lo[d]):(size(data,d)-1+hi[d])) : Colon(), ndims(data))
    return Array(@view data[axs...])
end

"""
    _body_mask(sim, nd, N_int) → Array{T}

Build a cell-centred multiplicative mask from the BDIM μ₀-kernel evaluated at
the cell-centre sdf: `mask = μ₀(sdf, ϵ)` ∈ `[0, 1]`. Zero deep inside the body
(`sdf ≤ -ϵ`), one in the fluid (`sdf ≥ +ϵ`), and a smooth cosine blend across
the band. Measured into a fresh buffer rather than read from `flow.σ` which
is scratch-space clobbered by `mom_step!`. The smooth kernel keeps the mask
numerically stable: FP differences in sdf between serial and parallel runs
produce linear changes in weight rather than 0→1 flips at a threshold.
"""
function _body_mask(a::AbstractSimulation, nd::Int, N_int::NTuple; kw...)
    σ = similar(a.flow.σ)
    WaterLily.measure_sdf!(σ, a.body, sum(a.flow.Δt); fastd²=Inf)
    σ_int = _interior(σ, nd, N_int; kw...)
    σ_int .= WaterLily.μ₀.(σ_int, eltype(σ_int)(a.ϵ))
    return σ_int
end

"""
    _apply_mask!(data, mask, nd) → data

Multiply `data` by `mask` elementwise, in place. `data` is either a scalar
field (same shape as `mask`) or a vector field (`mask`'s shape plus a trailing
component axis) — the mask is broadcast uniformly across components. `data`
must be an owned buffer (not a view into simulation state).
"""
function _apply_mask!(data::AbstractArray, mask::AbstractArray, nd::Int)
    if ndims(data) == nd               # scalar field
        data .*= mask
    else                                # vector field (spatial..., D)
        for c in 1:size(data, ndims(data))
            selectdim(data, ndims(data), c) .*= mask
        end
    end
    return data
end

"""
    save!(w::VTKWriter, a::AbstractSimulation)

Write the simulation data to VTK files.  Dispatches on `par_mode[]`:
  - Serial  → single `.vti` file
  - Parallel → per-rank `.vti` pieces + `.pvti` header (rank 0)

Attribute functions may return full-sized arrays (with `buff=1` ghost
cells) or already-stripped interior arrays; ghost layers are removed
automatically so the output contains interior cells only.
"""
function save!(w::VTKWriter, a::AbstractSimulation)
    _save!(w, a, WaterLily.par_mode[])
end

function _save!(w::VTKWriter, a::AbstractSimulation, ::WaterLily.Serial)
    k = w.count[1]; nd = ndims(a.flow.p); N_int = size(a.flow.p) .- 2
    pad = ntuple(d -> w.include_bc && !(d in a.flow.perdir) ? 1 : 0, nd)
    vtk = vtk_grid(w.dir_name*@sprintf("/%s_%06i", w.fname, k), [1:n+1 for n in N_int .+ 2 .* pad]...)
    mask = w.body_mask ? _body_mask(a, nd, N_int; lo=pad, hi=pad) : nothing
    for (name, func) in w.output_attrib
        data = _interior(func(a), nd, N_int; lo=pad, hi=pad)
        isnothing(mask) || _apply_mask!(data, mask, nd)
        vtk[name, VTKCellData()] = ndims(data) > nd ? components_first(data) : data
    end
    vtk_save(vtk); w.count[1]=k+1
    w.collection[round(sim_time(a),digits=4)]=vtk
end

function _save!(w::VTKWriter, a::AbstractSimulation, ::WaterLily.AbstractParMode)
    mpi_id = Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI")
    igg_id = Base.PkgId(Base.UUID("4d7a3746-15be-11ea-1130-334b0c4f5fa0"), "ImplicitGlobalGrid")

    if !haskey(Base.loaded_modules, mpi_id) || !haskey(Base.loaded_modules, igg_id)
        _save!(w, a, WaterLily.Serial()); return
    end

    MPIMod = Base.loaded_modules[mpi_id]
    IGG    = Base.loaded_modules[igg_id]

    comm   = MPIMod.COMM_WORLD
    me     = MPIMod.Comm_rank(comm)
    nprocs = MPIMod.Comm_size(comm)
    nprocs == 1 && (_save!(w, a, WaterLily.Serial()); return)

    g    = IGG.global_grid()
    nd   = sum(g.nxyz .> 1)               # active spatial dimensions
    nx   = ntuple(d -> g.nxyz[d] - g.overlaps[d], nd)   # interior cells per rank

    # Gather every rank's coords so all ranks build the same extents array.
    coords_mat = reshape(MPIMod.Allgather(Int32[g.coords[d] for d in 1:nd], comm), nd, nprocs)
    dims_proc  = ntuple(d -> Int(maximum(@view coords_mat[d, :])) + 1, nd)
    # BC layer per-dim (added only at non-periodic physical boundaries).
    bc = ntuple(d -> w.include_bc && !(d in a.flow.perdir) ? 1 : 0, nd)

    # Per-rank VTK point extents: leftmost/rightmost rank owns the BC cell;
    # middle ranks shift by `bc` to make room for it.
    extents = [ntuple(d -> begin
                 c = Int(coords_mat[d, r])
                 lo = (c == 0)              ? bc[d] : 0
                 hi = (c == dims_proc[d]-1) ? bc[d] : 0
                 s  = c*nx[d] + (c > 0)*bc[d]
                 (s+1):(s + nx[d] + lo + hi + 1)
               end, nd) for r in 1:nprocs]

    lo = ntuple(d -> Int(coords_mat[d,me+1])==0              ? bc[d] : 0, nd)
    hi = ntuple(d -> Int(coords_mat[d,me+1])==dims_proc[d]-1 ? bc[d] : 0, nd)

    k     = w.count[1]
    fname = joinpath(w.dir_name, @sprintf("%s_%06i", w.fname, k))
    pvtk  = pvtk_grid(fname, ntuple(d -> extents[me+1][d], nd)...; part=me+1, extents=extents)

    N_int = size(a.flow.p) .- 2
    mask  = w.body_mask ? _body_mask(a, nd, N_int; lo, hi) : nothing
    for (name, func) in w.output_attrib
        data = _interior(func(a), nd, N_int; lo, hi)
        isnothing(mask) || _apply_mask!(data, mask, nd)
        pvtk[name, VTKCellData()] = ndims(data) > nd ? components_first(data) : data
    end
    me == 0 ? (w.collection[round(sim_time(a), digits=4)] = pvtk) : vtk_save(pvtk)
    w.count[1] = k + 1
end

"""
    close(w::VTKWriter)

Closes the `VTKWriter`, this is required to write the collection file.
"""
close(w::VTKWriter)=(vtk_save(w.collection);nothing)
"""
    components_first(a::Array)

Permute the dimensions such that the u₁,u₂,(u₃) components of a vector field are the first dimensions and not the last
this is reqired for the vtk file.
"""
components_first(a::AbstractArray{T,N}) where {T,N} = permutedims(a,[N,1:N-1...])

end # module

module WaterLilyWriteVTKExt

using WriteVTK, WaterLily
import WaterLily: vtkWriter, save!, default_attrib, pvd_collection, vtk_attribs
using Printf: @sprintf
import Base: close

"""
    pvd_collection(fname;append=false)

Wrapper for a `paraview_collection` that returns a pvd file header
"""
pvd_collection(fname;append=false) = paraview_collection(fname;append=append)
"""
    VTKWriter(fname;attrib,dir,T,body_mask=false,include_bc=false)

Generates a `VTKWriter` that hold the collection name to which the `vtk` files are written.
The default attributes that are saved are the `Velocity` and the `Pressure` fields.
Custom attributes can be passed as `Dict{String,Function}` to the `attrib` keyword.
`body_mask=true` multiplies the fields by the BDIM mask `μ₀`; `include_bc=true` keeps the ghosts of non-periodic boundaries.
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
    mkpath(dir)
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

# attributes of `vtk_attribs`: each writes `flow.σ` and returns it
function _sdf_attrib(a::AbstractSimulation)
    σ, body, t, T = a.flow.σ, a.body, sum(a.flow.Δt), eltype(a.flow.σ)
    WaterLily.measure_sdf!(σ, body, t; fastd²=T(Inf))
    Rin = inside(σ)
    WaterLily.@loop σ[I] = (I in Rin ? σ[I] : WaterLily.sdf(body, loc(0,I,T), t)) over I ∈ CartesianIndices(σ)
    return σ
end
function _vorticity_attrib(a::AbstractSimulation)
    fill!(a.flow.σ, zero(eltype(a.flow.σ)))
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
    vtk_attribs(syms::Symbol...)

Attributes for `vtkWriter`, to merge with `default_attrib()`: `:sdf`, `:vorticity` (ω₃ in 2D, |ω| in 3D), `:lambda2`, `:rank`.
"""
function vtk_attribs(syms::Symbol...)
    d = Dict{String,Function}()
    for s in syms
        haskey(_attrib_lib, s) || error("unknown vtk attrib :$s. Available: $(collect(keys(_attrib_lib)))")
        name, func = _attrib_lib[s]; d[name] = func
    end
    return d
end

# `data` without its ghosts, but those of the sides with `lo[d]`/`hi[d]` = 1
function _interior(data::AbstractArray, nd::Int, N_int::NTuple;
                   lo::NTuple=ntuple(_->0, nd), hi::NTuple=ntuple(_->0, nd))
    size(data)[1:nd] == N_int && return data
    axs = ntuple(d -> d <= nd ? ((2-lo[d]):(size(data,d)-1+hi[d])) : Colon(), ndims(data))
    return Array(@view data[axs...])
end

# the BDIM mask μ₀ at the cell centres (ghosts included)
function _body_mask(a::AbstractSimulation, nd::Int, N_int::NTuple; kw...)
    σ, body, t, T = a.flow.σ, a.body, sum(a.flow.Δt), eltype(a.flow.σ)
    WaterLily.measure_sdf!(σ, body, t; fastd²=T((a.ϵ+1)^2))
    Rin = inside(σ)
    WaterLily.@loop σ[I] = (I in Rin ? σ[I] : WaterLily.sdf(body, loc(0,I,T), t)) over I ∈ CartesianIndices(σ)
    σ_int = _interior(σ, nd, N_int; kw...)
    σ_int .= WaterLily.μ₀.(σ_int, T(a.ϵ))
    return σ_int
end

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

function _write_attribs!(vtk, w::VTKWriter, a::AbstractSimulation, nd::Int, N_int::NTuple, lo::NTuple, hi::NTuple)
    mask = w.body_mask ? _body_mask(a, nd, N_int; lo, hi) : nothing
    for (name, func) in w.output_attrib
        data = _interior(func(a), nd, N_int; lo, hi)
        isnothing(mask) || _apply_mask!(data, mask, nd)
        vtk[name, VTKCellData()] = ndims(data) > nd ? components_first(data) : data
    end
end

"""
    save!(w::VTKWriter, sim<:AbstractSimulation)

Write the simulation data at time `sim_time(sim)` to a `vti` file (a piece of a `pvti` under MPI)
and add the file path to the collection file.
"""
function save!(w::VTKWriter, a::AbstractSimulation)
    w.include_bc && WaterLily.BC!(a.flow.u, a.flow.uBC, a.flow.exitBC, a.flow.perdir, sum(a.flow.Δt))
    _save!(w, a, WaterLily.par_mode[])
    fill!(a.flow.σ, zero(eltype(a.flow.σ))) # `CFL` reads all of σ
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
    haskey(Base.loaded_modules, mpi_id) || return _save!(w, a, WaterLily.Serial())

    MPIMod = Base.loaded_modules[mpi_id]
    comm = MPIMod.COMM_WORLD; me = MPIMod.Comm_rank(comm); nprocs = MPIMod.Comm_size(comm)
    nprocs == 1 && return _save!(w, a, WaterLily.Serial())

    nd = ndims(a.flow.p)
    nx = size(a.flow.p) .- 2
    off = WaterLily.global_offset(Val(nd), Int)
    coords_mat = reshape(MPIMod.Allgather(Int32[off[d] ÷ nx[d] for d in 1:nd], comm), nd, nprocs)
    dims_proc  = ntuple(d -> Int(maximum(@view coords_mat[d, :])) + 1, nd)
    bc = ntuple(d -> w.include_bc && !(d in a.flow.perdir) ? 1 : 0, nd)

    # the extent of each rank's piece, the first and last ranks with the ghosts of `include_bc`
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

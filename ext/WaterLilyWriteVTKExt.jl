module WaterLilyWriteVTKExt

using WriteVTK, WaterLily
import WaterLily: vtkWriter, save!, default_attrib, pvd_collection, save_parallel!
using Printf: @sprintf
import Base: close

"""
    pvd_collection(fname;append=false)

Wrapper for a `paraview_collection` that returns a pvd file header
"""
pvd_collection(fname;append=false) = paraview_collection(fname;append=append)
"""
    VTKWriter(fname;attrib,dir,T)

Generates a `VTKWriter` that hold the collection name to which the `vtk` files are written.
The default attributes that are saved are the `Velocity` and the `Pressure` fields.
Custom attributes can be passed as `Dict{String,Function}` to the `attrib` keyword.
"""
struct VTKWriter
    fname         :: String
    dir_name      :: String
    collection    :: WriteVTK.CollectionFile
    output_attrib :: Dict{String,Function}
    count         :: Vector{Int}
end
function vtkWriter(fname="WaterLily";attrib=default_attrib(),dir="vtk_data",T=Float32)
    !isdir(dir) && mkdir(dir)
    VTKWriter(fname,dir,pvd_collection(fname),attrib,[0])
end
function vtkWriter(fname,dir::String,collection,attrib::Dict{String,Function},k)
    VTKWriter(fname,dir,collection,attrib,[k])
end
"""
    default_attrib()

Returns a `Dict` containing the `name` and `bound_function` for the default attributes.
The `name` is used as the key in the `vtk` file and the `bound_function` generates the data
to put in the file. With this approach, any variable can be save to the vtk file.
"""
_velocity(a::AbstractSimulation) = a.flow.u |> Array;
_pressure(a::AbstractSimulation) = a.flow.p |> Array;
default_attrib() = Dict("Velocity"=>_velocity, "Pressure"=>_pressure)
"""
    save!(w::VTKWriter, sim<:AbstractSimulation)

Write the simulation data at time `sim_time(sim)` to a `vti` file and add the file path
to the collection file.
"""
function save!(w::VTKWriter, a::AbstractSimulation)
    k = w.count[1]; N=size(a.flow.p)
    vtk = vtk_grid(w.dir_name*@sprintf("/%s_%06i", w.fname, k), [1:n for n in N]...)
    for (name,func) in w.output_attrib
        # this seems bad, but I @benchmark it and it's the same as just calling func()
        vtk[name] = size(func(a))==N ? func(a) : components_first(func(a))
    end
    vtk_save(vtk); w.count[1]=k+1
    w.collection[round(sim_time(a),digits=4)]=vtk
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

"""
    save_parallel!(w::VTKWriter, a::AbstractSimulation)

MPI-aware parallel VTK writer.  Each rank writes its own `.vti` piece file and
rank 0 additionally writes the `.pvti` header, using WriteVTK's `pvtk_grid`.

Falls back to serial `save!` if MPI / ImplicitGlobalGrid are not loaded or if only
one process is running.

The custom attributes in `w.output_attrib` must return **interior-only** arrays
(ghost cells stripped).  Scalar fields have `ndim` dimensions; vector fields have
`ndim+1` dimensions with the last axis being components.
"""
function save_parallel!(w::VTKWriter, a::AbstractSimulation)
    mpi_id = Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI")
    igg_id = Base.PkgId(Base.UUID("4d7a3746-15be-11ea-1130-334b0c4f5fa0"), "ImplicitGlobalGrid")

    if !haskey(Base.loaded_modules, mpi_id) || !haskey(Base.loaded_modules, igg_id)
        save!(w, a); return
    end

    MPIMod = Base.loaded_modules[mpi_id]
    IGG    = Base.loaded_modules[igg_id]

    comm   = MPIMod.COMM_WORLD
    me     = MPIMod.Comm_rank(comm)
    nprocs = MPIMod.Comm_size(comm)
    nprocs == 1 && (save!(w, a); return)

    g    = IGG.global_grid()
    nd   = sum(g.nxyz .> 1)               # active spatial dimensions
    nx   = ntuple(d -> g.nxyz[d] - g.overlaps[d], nd)   # interior cells per rank

    # Gather every rank's coords so all ranks build the same extents array.
    my_coords  = Int32[g.coords[d] for d in 1:nd]
    all_coords = MPIMod.Allgather(my_coords, comm)
    coords_mat = reshape(all_coords, nd, nprocs)

    # Extents use VTK point indices (nx+1 points → nx cells).
    # Adjacent ranks overlap by 1 point so ParaView sees contiguous pieces.
    extents = [
        ntuple(d -> (Int(coords_mat[d,r]) * nx[d] + 1):(Int(coords_mat[d,r]) * nx[d] + nx[d] + 1), nd)
        for r in 1:nprocs
    ]

    k     = w.count[1]
    fname = joinpath(w.dir_name, @sprintf("%s_%06i", w.fname, k))

    pvtk = pvtk_grid(fname,
                     ntuple(d -> collect(Float32, extents[me+1][d]), nd)...;
                     part = me + 1, extents = extents)

    for (name, func) in w.output_attrib
        data = func(a)
        if ndims(data) > nd          # vector field (spatial..., D)
            pvtk[name, VTKCellData()] = components_first(data)
        else                         # scalar field
            pvtk[name, VTKCellData()] = data
        end
    end

    if me == 0
        w.collection[round(sim_time(a), digits=4)] = pvtk
    else
        vtk_save(pvtk)
    end
    w.count[1] = k + 1
end

end # module

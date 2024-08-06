module WaterLilyWriteVTKExt

if isdefined(Base, :get_extension)
    using WriteVTK
else
    using ..WriteVTK
end

using WaterLily
import WaterLily: vtkWriter, write!, default_attrib, pvd_collection
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
    extents       #:: Tuple{UnitRange}# cannot figure out what type to put here
end
function vtkWriter(fname="WaterLily";attrib=default_attrib(),dir="vtk_data",T=Float32,extents=[(1:1,1:1)])
    !isdir(dir) && mkdir(dir)
    VTKWriter(fname,dir,pvd_collection(fname),attrib,[0],extents)
end
function vtkWriter(fname,dir::String,collection,attrib::Dict{String,Function},k,extents)
    VTKWriter(fname,dir,collection,attrib,[k],extents)
end
"""
    default_attrib()

Returns a `Dict` containing the `name` and `bound_function` for the default attributes. 
The `name` is used as the key in the `vtk` file and the `bound_function` generates the data
to put in the file. With this approach, any variable can be save to the vtk file.
"""
_velocity(a::Simulation) = a.flow.u |> Array;
_pressure(a::Simulation) = a.flow.p |> Array;
default_attrib() = Dict("Velocity"=>_velocity, "Pressure"=>_pressure)
"""
    write!(w::VTKWriter, sim::Simulation)

Write the simulation data at time `sim_time(sim)` to a `vti` file and add the file path
to the collection file.
"""
function write!(w::VTKWriter,a::Simulation;N=size(a.flow.p))
    k = w.count[1]
    vtk = vtk_grid(w.dir_name*@sprintf("/%s_%06i", w.fname, k), [1:n for n in N]...)
    for (name,func) in w.output_attrib
        # this seems bad, but I @benchmark it and it's the same as just calling func()
        vtk[name] = size(func(a))==N ? func(a) : components_first(func(a))
    end
    vtk_save(vtk); w.count[1]=k+1
    w.collection[round(sim_time(a),digits=4)]=vtk
end
# parralel version of that file
function write!(w::VTKWriter,a::Simulation{D,T,S};N=size(inside(a.flow.p))) where {D,T,S<:MPIArray{T}}
    k,part = w.count[1], Int(me()+1)
    pvtk = pvtk_grid(w.dir_name*@sprintf("/%s_%06i", w.fname, k), w.extents[part];
                     part=part, extents=w.extents, ghost_level=2)
    for (name,func) in w.output_attrib
        # this seems bad, but I @benchmark it and it's the same as just calling func()
        pvtk[name] = size(func(a))==size(a.flow.p) ? func(a) : components_first(func(a))
    end
    vtk_save(pvtk); w.count[1]=k+1
    w.collection[round(sim_time(a),digits=4)]=pvtk
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
This is very anoying but is required to keep the file written neatly organised...
"""
function pvtk_grid(
        filename::AbstractString, args...;
        part, ismain = (part == 1), ghost_level = 0, kwargs...,
    )
    is_structured = WriteVTK._pvtk_is_structured(args...)
    nparts = WriteVTK._pvtk_nparts(is_structured; kwargs...)
    extents = WriteVTK._pvtk_extents(is_structured; kwargs...)

    # mkpath(filename)
    bname = basename(filename)
    prefix = filename #joinpath(filename, bname)
    fn = WriteVTK._serial_filename(part, nparts, prefix, "")

    vtk = let kws_vtk = WriteVTK._remove_parallel_kwargs(; kwargs...)
        kws = if extents === nothing
            kws_vtk
        else
            (; kws_vtk..., extent = extents[part])
        end
        vtk_grid(fn, args...; kws...)
    end

    pvtkargs = WriteVTK.PVTKArgs(part, nparts, ismain, ghost_level)
    xdoc = WriteVTK.XMLDocument()
    _, ext = splitext(vtk.path)
    path = filename * ".p" * ext[2:end]
    pvtk = WriteVTK.PVTKFile(pvtkargs, xdoc, vtk, path)
    _init_pvtk!(pvtk, extents)

    return pvtk
end
function _init_pvtk!(pvtk::WriteVTK.PVTKFile, extents)
    # Recover some data
    vtk = pvtk.vtk
    pvtkargs = pvtk.pvtkargs
    pgrid_type = "P" * vtk.grid_type
    npieces = pvtkargs.nparts
    pref, _ = splitext(pvtk.path)
    _, ext = splitext(vtk.path)
    prefix = basename(pref) #joinpath(pref, basename(pref))

    # VTKFile (root) node
    pvtk_root = WriteVTK.create_root(pvtk.xdoc, "VTKFile")
    WriteVTK.set_attribute(pvtk_root, "type", pgrid_type)
    WriteVTK.set_attribute(pvtk_root, "version", "1.0")
    if WriteVTK.IS_LITTLE_ENDIAN
        WriteVTK.set_attribute(pvtk_root, "byte_order", "LittleEndian")
    else
        WriteVTK.set_attribute(pvtk_root, "byte_order", "BigEndian")
    end

    # Grid node
    pvtk_grid = WriteVTK.new_child(pvtk_root, pgrid_type)
    WriteVTK.set_attribute(pvtk_grid, "GhostLevel", string(pvtkargs.ghost_level))

    # Pieces (i.e. Pointers to serial files)
    for piece ∈ 1:npieces
        pvtk_piece = WriteVTK.new_child(pvtk_grid, "Piece")
        fn = WriteVTK._serial_filename(piece, npieces, prefix, ext)
        WriteVTK.set_attribute(pvtk_piece, "Source", fn)

        # Add local extent if necessary
        if extents !== nothing
            WriteVTK.set_attribute(pvtk_piece, "Extent", WriteVTK.extent_attribute(extents[piece]))
        end
    end

    # Add whole extent if necessary
    whole_extent = WriteVTK.compute_whole_extent(extents)
    if whole_extent !== nothing
        WriteVTK.set_attribute(pvtk_grid, "WholeExtent", WriteVTK.extent_attribute(whole_extent))
    end

    # Getting original grid informations
    # Recover point type and number of components
    vtk_root = WriteVTK.root(vtk.xdoc)
    vtk_grid = WriteVTK.find_element(vtk_root, vtk.grid_type)

    # adding origin and spacing if necessary
    origin = WriteVTK.attribute(vtk_grid, "Origin")
    if origin !== nothing
        WriteVTK.set_attribute(pvtk_grid, "Origin", origin)
    end

    spacing = WriteVTK.attribute(vtk_grid, "Spacing")
    if spacing !== nothing
        WriteVTK.set_attribute(pvtk_grid, "Spacing", spacing)
    end

    # Getting original piece informations
    vtk_piece = WriteVTK.find_element(vtk_grid, "Piece")

    # If serial VTK has points
    vtk_points = WriteVTK.find_element(vtk_piece, "Points")
    if vtk_points !== nothing
        vtk_data_array = WriteVTK.find_element(vtk_points, "DataArray")
        point_type = WriteVTK.attribute(vtk_data_array, "type")
        Nc = WriteVTK.attribute(vtk_data_array, "NumberOfComponents")
        ## PPoints node
        pvtk_ppoints = WriteVTK.new_child(pvtk_grid, "PPoints")
        pvtk_pdata_array = WriteVTK.new_child(pvtk_ppoints, "PDataArray")
        WriteVTK.set_attribute(pvtk_pdata_array, "type", point_type)
        WriteVTK.set_attribute(pvtk_pdata_array, "Name", "Points")
        WriteVTK.set_attribute(pvtk_pdata_array, "NumberOfComponents", Nc)
    end

    # If serial VTK has coordinates
    vtk_coordinates = WriteVTK.find_element(vtk_piece, "Coordinates")
    if vtk_coordinates !== nothing
        pvtk_pcoordinates = WriteVTK.new_child(pvtk_grid, "PCoordinates")
        for c in WriteVTK.child_elements(vtk_coordinates)
            pvtk_pdata_array = WriteVTK.new_child(pvtk_pcoordinates, "PDataArray")
            WriteVTK.set_attribute(pvtk_pdata_array, "type", WriteVTK.attribute(c, "type"))
            WriteVTK.set_attribute(pvtk_pdata_array, "Name", WriteVTK.attribute(c, "Name"))
            WriteVTK.set_attribute(pvtk_pdata_array, "NumberOfComponents", WriteVTK.attribute(c, "NumberOfComponents"))
        end
    end

    pvtk
end

end # module

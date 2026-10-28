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
    VTKWriter(fname;attrib,dir,T)

Generates a `VTKWriter` that hold the collection name to which the `vtk` files are written.
The default attributes that are saved are the `Velocity` and the `Pressure` fields.
Custom attributes can be passed as `Dict{String,Function}` to the `attrib` keyword. Fields are written as cell data,
or as point data for an attribute `func=>:corner` at the cell corners (see `WaterLily.stagger`).
"""
struct VTKWriter
    fname         :: String
    dir_name      :: String
    collection    :: WriteVTK.CollectionFile
    output_attrib :: Dict{String}
    count         :: Vector{Int}
end
function vtkWriter(fname="WaterLily";attrib=default_attrib(),dir="vtk_data",T=Float32)
    !isdir(dir) && mkdir(dir)
    VTKWriter(fname,dir,pvd_collection(fname),attrib,[0])
end
function vtkWriter(fname,dir::String,collection,attrib::Dict{String},k)
    VTKWriter(fname,dir,collection,attrib,[k])
end
"""
    default_attrib()

Returns a `Dict` containing the `name` and `bound_function` for the default attributes.
The `name` is used as the key in the `vtk` file and the `bound_function` generates the data
to put in the file. With this approach, any variable can be save to the vtk file.
`Velocity` holds the staggered components for `load!`, written at the cell centers.
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
    vtk = vtk_grid(w.dir_name*@sprintf("/%s_%06i", w.fname, k), [0.5:n+0.5 for n in N]...) # cell I centered at I
    for (name,attr) in w.output_attrib
        func,at = attr isa Pair ? attr : (attr,:center)
        loc,data = vtk_data(func(a),at,N); vtk[name,loc] = data
    end
    vtk_save(vtk); w.count[1]=k+1
    w.collection[round(sim_time(a),digits=4)]=vtk
end
"""
    vtk_data(a,at,N)

Cell data for a field `a` at the cell centers, or point data padded to the `N.+1` points for a field at the cell
corners, the only locations `at` in a `vti` image. Vector components go first.
"""
function vtk_data(a,at,N)
    ndims(a)>length(N) && (a = components_first(a))
    s = WaterLily.stagger(at,length(N))
    all(iszero,s) && return VTKCellData(),a
    all(!iszero,s) || throw(ArgumentError("no $at location in a vti image"))
    b = zeros(eltype(a), size(a)[1:end-length(N)]..., (N.+1)...); b[CartesianIndices(a)] .= a
    return VTKPointData(),b
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

module WaterLilyWriteVTKExt

if isdefined(Base, :get_extension)
    using WriteVTK
else
    using ..WriteVTK
end

using WaterLily
import WaterLily: vtkWriter, write!
using Printf: @sprintf
import Base: close

"""
    vtkWriter(fname;attrib,dir,T)

Generates a `vtkWriter` that hold the collection name to which the `vtk` files are written.
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
    VTKWriter(fname,dir,paraview_collection(fname),attrib,[0])
end
"""
    default_attrib()

return a `Dict` containing the `name`` and `bound_function` for the default attributes. 
The `name` is used as the key in the `vtk` file and the `bound_function` generates the data
to put in the file. With this approach, any variable can be save to the vtk file.
"""
_velocity(a::Simulation) = a.flow.u |> Array;
_pressure(a::Simulation) = a.flow.p |> Array;
default_attrib() = Dict("Velocity"=>_velocity, "Pressure"=>_pressure)
"""
    write!(w::vtkWriter, sim::Simulation)

Write the simulation data at time `sim_time(sim)` to a `vti` file and add the file path
to the collection file.
"""
function write!(w::VTKWriter,a::Simulation)
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
    close(w::vtkWriter)

closes the `vtkWriter`, this is required to write the collection file.
"""
close(w::VTKWriter)=(vtk_save(w.collection);nothing)
"""
    components_first(a::Array)

Permute the dimensions such that the u₁,u₂,(u₃) components of a vector field are the first dimensions and not the last
this is reqired for the vtk file.
"""
components_first(a::AbstractArray{T,N}) where {T,N} = permutedims(a,[N,1:N-1...])

end # module

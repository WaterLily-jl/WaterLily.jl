using WriteVTK
using Printf: @sprintf
"""
    vtk_grid(name;attrib,T)

Generates a `vtkWriter` that hold the collection name to which the `vtk` files are written.
The default attributes that are saved are the `Velocity` and the `Pressure` fields.
Custom attributes can be passed as `Dict{String,Function}` to the `attrib` keyword.    
"""
struct vtkWriter
    fname::String
    collection::WriteVTK.CollectionFile
    output_attrib::Dict{String,Function}
    count::Vector{Int}
    function vtkWriter(fname="WaterLily";attrib=default_attrib(),T=Float32)
        new(fname,paraview_collection(fname),attrib,[0])
    end
end
"""
    default_attrib()

return a `Dict` containing the name and bound funtion for the default attributes. 
The name is used as the key in the `vtk` file and the function generates the data
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
function write!(w::vtkWriter, sim::Simulation)
    k = w.count[1]; N=size(sim.flow.p)
    vtk = vtk_grid(@sprintf("%s_%02i", w.fname, k), [1:n for n in N]...)
    for (name,func) in w.output_attrib
        # this seems bad, but I @benchmark it and it's the same as just calling func()
        vtk[name] = size(func(sim))==N ? func(sim) : components_first(func(sim))
    end
    vtk_save(vtk); w.count[1]=k+1
    w.collection[round(sim_time(sim),digits=4)]=vtk
end
"""
    close(w::vtkWriter)

closes the `vtkWriter`, this is required to write the collection file.
"""
Base.close(w::vtkWriter)=(vtk_save(w.collection);nothing)
"""
    components_first(a::Array)

Permute the dimensions such that the u₁,u₂,(u₃) components of a vector field are the first dimensions and not the last
this is reqired for the vtk file.
"""
function components_first(a::Array)
    N=length(size(a)); p=[N,1:N-1...]
    return permutedims(a,p)
end
using WriteVTK, ReadVTK
using Printf: @sprintf
"""
    vtkWriter(fname;attrib,dir,T)

Generates a `vtkWriter` that hold the collection name to which the `vtk` files are written.
The default attributes that are saved are the `Velocity` and the `Pressure` fields.
Custom attributes can be passed as `Dict{String,Function}` to the `attrib` keyword.    
"""
struct vtkWriter
    fname         :: String
    dir_name      :: String
    collection    :: WriteVTK.CollectionFile
    output_attrib :: Dict{String,Function}
    count         :: Vector{Int}
    function vtkWriter(fname="WaterLily";attrib=default_attrib(),dir="vtk_data",T=Float32)
        !isdir(dir) && mkdir(dir)
        new(fname,dir,paraview_collection(fname),attrib,[0])
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
function write!(w::vtkWriter,a::Simulation)
    k = w.count[1]; N=size(a.flow.p)
    vtk = vtk_grid(w.dir_name*@sprintf("/%s_%02i", w.fname, k), [1:n for n in N]...)
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
Base.close(w::vtkWriter)=(vtk_save(w.collection);nothing)
"""
    components_first(a::Array)

Permute the dimensions such that the u₁,u₂,(u₃) components of a vector field are the first dimensions and not the last
this is reqired for the vtk file.
"""
components_first(a::AbstractArray{T,N}) where {T,N} = permutedims(a,[N,1:N-1...])
"""
    components_last(a::Array)

Permute the dimensions such that the u₁,u₂,(u₃) components of a vector field are the last dimensions and not the first
this is reqired for the vtk file.
"""
components_last(a::AbstractArray{T,N}) where {T,N} = permutedims(a,[2:N...,1])
"""
    restart_sim!(a::Simulation;fname::String="WaterLily.pvd")

Restart a simulation from a pvd collection file, using the last saved vtk file
The velocity and pressure field of sim are overwriten by the one in the vtk file.
The time step is also updated to match the time step of the vtk file, such that
the simulation can be restarted and continued.
"""
function restart_sim!(a::Simulation;fname::String="WaterLily.pvd")
    vtk = VTKFile(PVDFile(fname).vtk_filenames[end])
    extent = ReadVTK.get_whole_extent(vtk)[2:2:end]
    # check dimensions match
    text = "The dimensions of the simulation do not match the dimensions of the vtk file"
    @assert extent.+1 == collect(size(a.flow.p)) text
    # fill the arrays for pressure and velocity
    point_data = ReadVTK.get_point_data(vtk)
    a.flow.p .= Array(get_data_reshaped(point_data["Pressure"]))
    a.flow.u .= components_last(Array(get_data_reshaped(point_data["Velocity"])))
    # reset time to work with the new time step
    a.flow.Δt[end] = PVDFile(fname).timesteps[end]*a.L/a.U
    push!(a.flow.Δt,WaterLily.CFL(a.flow))
end
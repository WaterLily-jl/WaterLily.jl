module WaterLilyReadVTKExt

if isdefined(Base, :get_extension)
    using ReadVTK
else
    using ..ReadVTK
end

using WaterLily
import WaterLily: restart_sim!

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
function restart_sim!(a::AbstractSimulation;fname::String="WaterLily.pvd",attrib=default_attrib())
    vtk = VTKFile(PVDFile(fname).vtk_filenames[end])
    extent = filter(!iszero,ReadVTK.get_whole_extent(vtk)[2:2:end]);
    # check dimensions match
    text = "The dimensions of the simulation do not match the dimensions of the vtk file"
    @assert extent.+1 == collect(size(a.flow.p)) text
    # fill the arrays for pressure and velocity
    point_data = ReadVTK.get_point_data(vtk)
    copyto!(a.flow.p, WaterLily.squeeze(Array(get_data_reshaped(point_data["Pressure"]))));
    copyto!(a.flow.u, WaterLily.squeeze(components_last(Array(get_data_reshaped(point_data["Velocity"])))));
    # reset time to work with the new time step
    a.flow.Δt[end] = PVDFile(fname).timesteps[end]*a.L/a.U
    push!(a.flow.Δt,WaterLily.CFL(a.flow))
    # return a writer if needed
    k = length(PVDFile(fname).timesteps)
    vtkWriter(split(fname,".pvd")[1],PVDFile(fname).directories[1],
              WaterLily.pvd_collection(fname;append=true),attrib,k)
end

end # module
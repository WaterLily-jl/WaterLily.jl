module WaterLilyReadVTKExt

using ReadVTK, WaterLily
import WaterLily: load!

"""
    components_last(a::Array)

Permute the dimensions such that the u₁,u₂,(u₃) components of a vector field are the last dimensions and not the first
this is reqired for the vtk file.
"""
components_last(a::AbstractArray{T,N}) where {T,N} = permutedims(a,[2:N...,1])

"""
    load!(a::AbstractSimulation, ::Val{:pvd}; kwargs...)

Restart a simulation from a pvd collection file, using the last saved vtk file.
The velocity and pressure field of `a::AbstractSimulation`` are overwriten by the one in the vtk file.
The time step is also updated to match the time step of the vtk file, such that the simulation can be restarted and continued.
Keyword arguments considered are `fname="WaterLily.pvd"` and `attrib=default_attrib()`.
"""
function load!(a::AbstractSimulation, ::Val{:pvd}; kwargs...)
    fname = get(Dict(kwargs), :fname, "WaterLily.pvd")
    attrib = get(Dict(kwargs), :attrib, default_attrib())
    vtk = VTKFile(PVDFile(fname).vtk_filenames[end])
    extent = filter(!iszero,ReadVTK.get_whole_extent(vtk)[2:2:end]);
    # VTK stores interior cells only (buff=2 ghost layers stripped on save).
    Ni = collect(size(a.flow.p) .- 4)
    text = "The dimensions of the simulation do not match the dimensions of the vtk file."
    @assert extent == Ni text
    # load into the interior region; ghosts are refreshed by BC!
    cell_data = ReadVTK.get_cell_data(vtk)
    pressure = get(Dict(kwargs), :pressure, "Pressure")
    velocity = get(Dict(kwargs), :velocity, "Velocity")
    p_data = WaterLily.squeeze(Array(get_data_reshaped(cell_data[pressure], cell_data=true)))
    u_data = WaterLily.squeeze(components_last(Array(get_data_reshaped(cell_data[velocity], cell_data=true))))
    nd = length(Ni)
    p_int = ntuple(d -> 3:size(a.flow.p, d)-2, nd)
    u_int = ntuple(d -> d <= nd ? (3:size(a.flow.u, d)-2) : Colon(), ndims(a.flow.u))
    copyto!(view(a.flow.p, p_int...), p_data)
    copyto!(view(a.flow.u, u_int...), u_data)
    WaterLily.BC!(a.flow.u, a.flow.uBC, a.flow.exitBC, a.flow.perdir)
    WaterLily.comm!(a.flow.p, a.flow.perdir)
    # reset time to work with the new time step
    a.flow.Δt[end] = PVDFile(fname).timesteps[end]*a.L/a.U
    push!(a.flow.Δt,WaterLily.CFL(a.flow))
    # return a writer if needed
    k = length(PVDFile(fname).timesteps)
    vtkWriter(split(fname,".pvd")[1],PVDFile(fname).directories[1],
              WaterLily.pvd_collection(fname;append=true),attrib,k)
end

end # module
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
    last_vtk = PVDFile(fname).vtk_filenames[end]
    # Parallel: the .pvd points at a .pvti; each rank reads its own piece
    # (named <base>/<base>_<rank+1>.vti by `pvtk_grid` in the writer).
    if endswith(last_vtk, ".pvti")
        base = splitext(last_vtk)[1]
        last_vtk = joinpath(base, basename(base) * "_$(WaterLily.mpi_rank()+1).vti")
    end
    vtk = VTKFile(last_vtk)
    # Cell count and 0-indexed start per active dim. `start[d]==0` ⇒ this piece
    # owns the lo BC cell (relevant only when saved with `include_bc=true`).
    ext   = ReadVTK.get_whole_extent(vtk)
    pairs = [(ext[2d-1], ext[2d] - ext[2d-1]) for d in 1:length(ext)÷2 if ext[2d] != ext[2d-1]]
    start = first.(pairs); cells = last.(pairs)
    Ni    = collect(size(a.flow.p) .- 2)
    extra = cells .- Ni  # 0 = no BC layer; 1 = one side BC; 2 = both sides BC.
    @assert all(0 .≤ extra .≤ 2) "Dimensions of the simulation do not match the vtk file."
    # If the file carries BC layers (include_bc=true at save), strip them: BC!
    # below restores the ghost layer identically. lo BC is present iff this
    # piece starts at the global origin in that dim.
    nd     = length(Ni)
    lo_pad = ntuple(d -> (start[d] == 0 && extra[d] ≥ 1) ? 1 : 0, nd)
    hi_pad = ntuple(d -> extra[d] - lo_pad[d], nd)
    src    = ntuple(d -> (lo_pad[d]+1):(lo_pad[d]+Ni[d]), nd)
    cell_data = ReadVTK.get_cell_data(vtk)
    pressure = get(Dict(kwargs), :pressure, "Pressure")
    velocity = get(Dict(kwargs), :velocity, "Velocity")
    p_data = WaterLily.squeeze(Array(get_data_reshaped(cell_data[pressure], cell_data=true)))
    u_data = WaterLily.squeeze(components_last(Array(get_data_reshaped(cell_data[velocity], cell_data=true))))
    p_int  = ntuple(d -> 2:size(a.flow.p, d)-1, nd)
    u_int  = ntuple(d -> d <= nd ? (2:size(a.flow.u, d)-1) : Colon(), ndims(a.flow.u))
    u_src  = ntuple(d -> d <= nd ? src[d] : Colon(), ndims(u_data))
    copyto!(view(a.flow.p, p_int...), view(p_data, src...))
    copyto!(view(a.flow.u, u_int...), view(u_data, u_src...))
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
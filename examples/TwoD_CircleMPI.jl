#mpiexecjl --project=examples/ -n 4 julia TwoD_CircleMPI.jl

using WaterLily
using WriteVTK
using MPI
using StaticArrays
using Printf: @sprintf
# include("../WaterLilyMPI.jl") # this uses the old functions

function WriteVTK.pvtk_grid(
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
    WriteVTK._init_pvtk!(pvtk, extents)

    return pvtk
end


function WriteVTK._init_pvtk!(pvtk::WriteVTK.PVTKFile, extents)
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

components_first(a::AbstractArray{T,N}) where {T,N} = permutedims(a,[N,1:N-1...])
# function write!(w,a::Simulation{D,T,S};N=size(inside(sim.flow.p))) where {D,T,S<:MPIArray{T}}
function write!(w,a::Simulation;N=size(inside(sim.flow.p))) 
    k = w.count[1]
    xs = Tuple(ifelse(x==0,1,x+3):ifelse(x==0,n+4,n+x+6) for (n,x) in zip(N,grid_loc()))
    extents = MPI.Allgather(xs, mpi_grid().comm)
    part = Int(me()+1)
    pvtk = pvtk_grid(w.dir_name*@sprintf("/%s_%06i", w.fname, k), extents[part]; part=part, extents=extents, ghost_level=0)
    for (name,func) in w.output_attrib
        # this seems bad, but I @benchmark it and it's the same as just calling func()
        pvtk[name] = size(func(a))==size(sim.flow.p) ? func(a) : components_first(func(a))
    end
    vtk_save(pvtk); w.count[1]=k+1
    w.collection[round(sim_time(a),digits=4)]=pvtk
end


"""Flow around a circle"""
function circle(n,m,center,radius;Re=250,U=1,psolver=Poisson,mem=Array)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, mem=mem, psolver=psolver)
end

# make a writer with some attributes
velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
                        a.flow.σ |> Array;)
vorticity(a::Simulation) = (@inside a.flow.σ[I] = 
                            WaterLily.curl(3,I,a.flow.u)*a.L/a.U;
                            a.flow.σ |> Array;)
_vbody(a::Simulation) = a.flow.V |> Array;
mu0(a::Simulation) = a.flow.μ₀ |> Array;

custom_attrib = Dict(
    "u" => velocity,
    "p" => pressure,
    "d" => _body,
    "ω" => vorticity,
    "v" => _vbody,
    "μ₀" => mu0
)# this maps what to write to the name in the file

WaterLily.grid_loc() = mpi_grid().global_loc

# local grid size
nx = 2^6
ny = 2^6

# init the MPI grid and the simulation
r = init_mpi((nx,ny))
sim = circle(nx,ny,SA[ny/2,ny/2+2],nx/8;mem=MPIArray) #use MPIArray to use extension

wr = vtkWriter("WaterLily-circle-2";attrib=custom_attrib,dir="vtk_data")
for _ in 1:5
    sim_step!(sim,sim_time(sim)+1.0,verbose=true)
    write!(wr,sim)
end
close(wr)
finalize_mpi()
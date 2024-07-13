using MPI
using WaterLily
using WriteVTK
using Printf: @sprintf

include("WaterLilyMPI.jl")

components_first(a::AbstractArray{T,N}) where {T,N} = permutedims(a,[N,1:N-1...])

"""Flow around a circle"""
function circle(n,m,center,radius;Re=250,U=1,psolver=Poisson)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, psolver=psolver)
end

function write!(w,a::Simulation)
    k = w.count[1]; N = size(inside(sim.flow.p))
    xs = Tuple(ifelse(x==0,1,x+3):ifelse(x==0,n+4,n+x+6) for (n,x) in zip(N,grid_loc()))
    extents = MPI.Allgather(xs, mpi_grid().comm)
    part = Int(me()+1)
    pvtk = pvtk_grid(@sprintf("%s_%06i", w.fname, k), extents[part]; part=part, extents=extents, ghost_level=2)
    for (name,func) in w.output_attrib
        # this seems bad, but I @benchmark it and it's the same as just calling func()
        pvtk[name] = size(func(a))==size(sim.flow.p) ? func(a) : components_first(func(a))
    end
    vtk_save(pvtk); w.count[1]=k+1
    w.collection[round(sim_time(a),digits=4)]=pvtk
end

# local grid size
nx = 2^6
ny = 2^6

# init the MPI grid and the simulation
r = init_mpi((nx,ny))
sim = circle(nx,ny,SA[ny/2,ny/2],nx/8)

wr = vtkWriter("fields";attrib=default_attrib(),dir="vtk_data")
for _ in 1:5
    sim_step!(sim,sim_time(sim)+1.0,verbose=true)
    write!(wr,sim)
end
close(wr)

finalize_mpi()
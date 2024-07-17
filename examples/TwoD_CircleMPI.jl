#mpiexecjl --project= -n 4 julia TwoD_CircleMPI.jl

using WaterLily
using WriteVTK
using MPI
using StaticArrays
using Printf: @sprintf
# include("../WaterLilyMPI.jl") # this uses the old functions

# make a writer with some attributes, now we need to apply the BCs when writting
velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body);
                        a.flow.σ |> Array;)
vorticity(a::Simulation) = (@inside a.flow.σ[I] = 
                            WaterLily.curl(3,I,a.flow.u)*a.L/a.U;
                            WaterLily.perBC!(a.flow.σ,());
                            a.flow.σ |> Array;)
_vbody(a::Simulation) = a.flow.V |> Array;
mu0(a::Simulation) = a.flow.μ₀ |> Array;
ranks(a::Simulation) = (a.flow.σ.=0; 
                        @inside a.flow.σ[I] = me()+1;
                        WaterLily.perBC!(a.flow.σ,());
                        a.flow.σ |> Array;)

custom_attrib = Dict(
    "u" => velocity,
    "p" => pressure,
    "d" => _body,
    "ω" => vorticity,
    "v" => _vbody,
    "μ₀" => mu0,
    "rank" => ranks
)# this maps what to write to the name in the file

"""Flow around a circle"""
function circle(dims,center,radius;Re=250,U=1,psolver=MultiLevelPoisson,mem=Array)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation(dims, (U,0), radius; ν=U*radius/Re, body, mem=mem, psolver=psolver)
end

# last one standing...
WaterLily.grid_loc() = mpi_grid().global_loc

# local grid size
L = 2^6

# init the MPI grid and the simulation
r = init_mpi((L,L))
sim = circle((L,L),SA[L/2,L/2+2],L/8;mem=MPIArray) #use MPIArray to use extension

wr = vtkWriter("WaterLily-circle-2";attrib=custom_attrib,dir="vtk_data",
               extents=get_extents(sim.flow.p))
for _ in 1:50
    sim_step!(sim,sim_time(sim)+1.0,verbose=true)
    write!(wr,sim)
end
close(wr)
finalize_mpi()
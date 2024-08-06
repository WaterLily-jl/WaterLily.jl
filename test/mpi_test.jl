# test/MPI_test.jl
using WaterLily,MPI
using StaticArrays
using FileIO,JLD2

WaterLily.L₂(ml::MultiLevelPoisson) = WaterLily.L₂(ml.levels[1])
WaterLily.L∞(ml::MultiLevelPoisson) = WaterLily.L∞(ml.levels[1])

"""Flow around a circle"""
function circle(dims,center,radius;Re=250,U=1,psolver=MultiLevelPoisson,mem=MPIArray)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation(dims, (U,0), radius; ν=U*radius/Re, body, mem=mem, psolver=psolver)
end

# init the MPI grid and the simulation
L = 2^6
r = init_mpi((L,L))
sim = circle((L,L),SA[L/2,L/2+2],L/8;mem=MPIArray)

# check global coordinates
x1 = global_loc(0,CartesianIndex(3,3))
x2 = global_loc(CartesianIndex(3,3,1))
save("global_loc_$(me()).jld2", "data", [x1,x2])

# first we check simple rank matrix
sim.flow.σ .= NaN
sim.flow.σ[inside(sim.flow.σ)] .= me() #reshape(collect(1:length(inside(sim.flow.σ))),size(inside(sim.flow.σ)))
save("sigma_1_$(me()).jld2", "data", sim.flow.σ)
# updating halos
WaterLily.perBC!(sim.flow.σ,())
save("sigma_2_$(me()).jld2", "data", sim.flow.σ)

# test global sdf
sim.flow.σ .= NaN
# check that the measure uses the correct loc function
measure_sdf!(sim.flow.σ,sim.body,0.0)
save("sdf_3_$(me()).jld2", "data", sim.flow.σ)
# updating the halos here
WaterLily.perBC!(sim.flow.σ,())
save("sdf_4_$(me()).jld2", "data", sim.flow.σ)

# test on vector field
measure!(sim,0.0)
save("mu0_1_$(me()).jld2", "data", sim.flow.μ₀[:,:,1])
save("mu0_2_$(me()).jld2", "data", sim.flow.μ₀[:,:,2])

#try a momentum step
sim = circle((L,L),SA[L/2,L/2+2],L/8;mem=MPIArray)
mom_step!(sim.flow,sim.pois)
me()==0 && println("mom_step! with $(sim.pois.n) MG iters $(typeof(sim.pois))")
save("mom_step_$(me())_p.jld2","data",sim.flow.p)
save("mom_step_$(me())_u1.jld2","data",sim.flow.u[:,:,1])
save("mom_step_$(me())_u2.jld2","data",sim.flow.u[:,:,2])

# test norm functions
sim.pois.levels[1].r .= 0.0
me() == 1 && (sim.pois.levels[1].r[32,32] = 123.456789) # make this the only non-zero element
Linf = WaterLily.L∞(sim.pois)
L2 = WaterLily.L₂(sim.pois)
save("norm_$(me()).jld2", "data", [Linf,L2])
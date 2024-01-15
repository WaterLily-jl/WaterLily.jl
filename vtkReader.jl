using WaterLily
using StaticArrays
using Test

function circle(p=4;Re=250,mem=Array,U=1)
    # Define simulation size, geometry dimensions, viscosity
    L=2^p
    center,r = SA[3L,3L,0], L/2
    # make a body
    body = AutoBody() do xyz,t
        x,y,z = xyz - center
        √sum(abs2,SA[x,y,0])-r
    end
    Simulation((8L,6L,16),(U,0,0),L;ν=U*L/Re,body,mem)
end
# make a simulation
sim = circle();
# make a vtk writer
wr = vtkWriter("test_vtk_reader";dir="TEST_DIR")
sim_step!(sim,1)
write!(wr, sim)
close(wr)

# re start the sim from a paraview file
sim2 = circle();
restart_sim!(sim2; fname="test_vtk_reader.pvd")

@test all(sim.flow.p .== sim2.flow.p)
@test all(sim.flow.u .== sim2.flow.u)
@test all(sim.flow.μ₀ .== sim2.flow.μ₀)
@test sim.flow.Δt[end] == sim2.flow.Δt[end]
@test abs(sim_time(sim)-sim_time(sim2))<1e-3

# clean-up
@test_nowarn rm("TEST_DIR", recursive = true)
@test_nowarn rm("test_vtk_reader.pvd")

# # intialize
# t₀ = sim_time(sim)
# duration = 10
# tstep = 0.1

# # step and write
# @time for tᵢ in range(t₀,t₀+duration;step=tstep)
#     # update until time tᵢ in the background
#     sim_step!(sim,tᵢ,remeasure=false)

#     # print time step
#     println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
# end

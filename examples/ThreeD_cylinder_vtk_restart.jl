using WaterLily
using ReadVTK, WriteVTK
using StaticArrays

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

# make a simulation on the CPU
sim = circle();
# make a vtk writer
wr = vtkWriter("TwoD_circle_vtk_restart")
# sim for a bit, write and close
sim_step!(sim,1); write!(wr, sim); close(wr)

# re-start the sim from a paraview file but on the GPU this time
import CUDA
restart = circle(;mem=CUDA.CuArray);
wr2 = restart_sim!(restart; fname="TwoD_circle_vtk_restart.pvd")

# intialize
t₀ = sim_time(restart); duration = 10; tstep = 0.1

# step and write for a longer time
@time for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    sim_step!(restart,tᵢ,remeasure=false)
    # write again to the "TwoD_circle_vtk_restart.pvd" file
    write!(wr2, restart)
    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(restart.flow.Δt[end],digits=3))
end
close(wr2)

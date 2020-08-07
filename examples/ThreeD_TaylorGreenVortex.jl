using WaterLily
using LinearAlgebra: norm2
using Makie

function flowdata(sim)
    @inside sim.flow.σ[I] = WaterLily.ω_mag(I,sim.flow.u)*sim.L/sim.U
    return @view sim.flow.σ[2:end-1,2:end-1,2:end-1]
end
function TGV_video(p=6,Re=1e5,Δprint=0.1,nprint=100)
    # Define vortex size, velocity, viscosity
    L = 2^p; U = 1; ν = U*L/Re

    # Taylor-Green-Vortex initial velocity field
    u = apply(L+2,L+2,L+2,3) do i,vx
        x,y,z = @. (vx-1.5)*π/L                # scaled coordinates
        i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x
        i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y
        return 0.                              # u_z
    end

    # Initialize simulation
    c = ones(L+2,L+2,L+2,3)  # no immersed solids
    a = Flow(u,c,zeros(3),ν=ν)
    b = MultiLevelPoisson(c)
    sim = Simulation(U,L,a,b)

    # plot the vorticity modulus
    scene = Scene(backgroundcolor = :black)
    scene = volume!(scene,flowdata(sim),colorrange=(π,4π),algorithm = :absorption)
    vol_plot = scene[end]

    # Plot flow evolution
    tprint = 0.0
    record(scene,"file.mp4",1:nprint,framerate=24,compression=5) do i
        tprint += Δprint
        sim_step!(sim,tprint)
        println("video ",round(Int,i/nprint*100),"% complete")
        vol_plot[1] = flowdata(sim)
    end
    return sim,scene
end

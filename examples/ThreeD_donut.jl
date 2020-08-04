using WaterLily
using LinearAlgebra: norm2
using Makie

function donut_sim(p=6,Re=1e3)
    # Define simulation size, velocity, viscosity
    n,U = 2^p, [1, 0, 0]
    center,R,r = [n/2,n/2,n/2], n/4, n/16
    ν = norm2(U)*R/Re

    # Apply signed distance function for a torus
    c = BDIM_coef(2n+2,n+2,n+2,3) do xyz  #
        x,y,z = xyz - center
        norm2([x,norm2([y,z])-R])-r
    end

    # Initialize Flow, Poisson and make struct
    u = zeros(2n+2,n+2,n+2,3)
    a = Flow(u,c,U,ν=ν)
    b = MultiLevelPoisson(c)
    Simulation(norm2(U),R,a,b),center
end

function flowdata(sim)
    @inside sim.a.σ[I] = WaterLily.ω_θ(I,[1,0,0],center,sim.a.u)*sim.L/sim.U
    @view sim.a.σ[2:end-1,2:end-1,2:end-1]
end
function geomdata(sim)
    @inside sim.a.σ[I] = sum(sim.a.μ₀[I,i]+sim.a.μ₀[I+δ(i,I),i] for i=1:3)
    @view sim.a.σ[2:end-1,2:end-1,2:end-1]
end

function make_video!(sim::Simulation;name="file.mp4",verbose=true,t₀=0.0,Δprint=0.1,nprint=24*3)
    # plot the geometry and flow
    scene = contour(geomdata(sim),levels=[0.5])
    scene = contour!(scene,flowdata(sim),levels=[-7,7],
                     colormap=:balance,alpha=0.2,colorrange=[-7,7])
    scene_data = scene[end]

    # Plot flow evolution
    tprint = t₀+WaterLily.sim_time(sim)
    record(scene,name,1:nprint,compression=5) do i
        tprint+=Δprint
        sim_step!(sim,tprint;verbose)
        println("video ",round(Int,i*100/nprint),"% complete")
        scene_data[1] = flowdata(sim)
    end
    return scene
end

#donut,center = donut_sim();
#scene = make_video!(donut);

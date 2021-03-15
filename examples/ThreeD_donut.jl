using WaterLily
using LinearAlgebra: norm2
using Makie

function donut_sim(;p=6,Re=1e3)
    # Define simulation size, geometry dimensions, viscosity
    n = 2^p
    center,R,r = [n/2,n/2,n/2], n/4, n/16
    ν = R/Re
    @show R,ν

    # Apply signed distance function for a torus
    body = AutoBody() do xyz,t
        x,y,z = xyz - center
        norm2([x,norm2([y,z])-R])-r
    end

    # Initialize simulation
    Simulation((2n+2,n+2,n+2),[1.,0.,0.],R;ν,body),center
end

function flowdata(sim)
    @inside sim.flow.σ[I] = WaterLily.ω_θ(I,[1,0,0],center,sim.flow.u)*sim.L/sim.U
    @view sim.flow.σ[2:end-1,2:end-1,2:end-1]
end
function geomdata(sim)
    @inside sim.flow.σ[I] = sum(sim.flow.μ₀[I,i]+sim.flow.μ₀[I+δ(i,I),i] for i=1:3)
    @view sim.flow.σ[2:end-1,2:end-1,2:end-1]
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

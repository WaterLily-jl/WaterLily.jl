using WaterLily
using LinearAlgebra: norm2
include("ThreeD_Plots.jl")

function donut(p=6,Re=1e3)
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

function ω_θ_data(sim)
    @inside sim.flow.σ[I] = WaterLily.ω_θ(I,[1,0,0],center,sim.flow.u)*sim.L/sim.U
    @view sim.flow.σ[2:end-1,2:end-1,2:end-1]
end
function body_data(sim)
    @inside sim.flow.σ[I] = sum(sim.flow.μ₀[I,i]+sim.flow.μ₀[I+δ(i,I),i] for i=1:3)
    @view sim.flow.σ[2:end-1,2:end-1,2:end-1]
end

sim,center = donut();
sim,fig = contour_video!(sim,ω_θ_data,body_data,name="donut.mp4",duration=10);

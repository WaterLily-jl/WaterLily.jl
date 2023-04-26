using WaterLily
using StaticArrays
function donut(p=6;Re=1e3,mem=Array,U=1)
    # Define simulation size, geometry dimensions, viscosity
    n = 2^p
    center,R,r = SA[n/2,n/2,n/2], n/4, n/16
    ν = U*R/Re

    # Apply signed distance function for a torus
    norm2(x) = √sum(abs2,x)
    body = AutoBody() do xyz,t
        x,y,z = xyz - center
        norm2(SA[x,norm2(SA[y,z])-R])-r
    end

    # Initialize simulation and return center for flow viz
    Simulation((2n,n,n),(U,0,0),R;ν,body,mem),center
end

import CUDA
@assert CUDA.functional()
sim,center = donut(mem=CUDA.CuArray);
#sim,center = donut(mem=Array); # if you don't have a CUDA GPU

dat = sim.flow.σ[inside(sim.flow.σ)] |> Array;
function ω_θ!(dat,sim,center=center)
    dt, a = sim.L/sim.U, sim.flow.σ
    @inside a[I] = WaterLily.ω_θ(I,(1,0,0),center,sim.flow.u)*dt
    copyto!(dat,a[inside(a)]) 
end

include("ThreeD_Plots.jl")
@time makie_video!(sim,dat,ω_θ!,name="donut.mp4",duration=10,step=0.25) do obs
    contour(obs, levels=[-5,5], colormap=:balance)
end

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

using KernelAbstractions
using CUDA
CUDA.allowscalar(false)
using GLMakie
begin
    # Set-up GPU simulation
    sim,center = donut(mem=CuArray);

    # Plot donut surface contour from CPU
    d = sim.flow.σ |> Array;
    fig, ax, plt1 = contour(d[inside(d)], levels=[0.5])

    # Set up observable and fill it with ω_θ
    dat = d[inside(d)] |> Observable;
    function ω_θ!(dat,d,sim,center)
        dt = sim.L/sim.U
        @inside sim.flow.σ[I] = WaterLily.ω_θ(I,(1,0,0),center,sim.flow.u)*dt
        copyto!(d,sim.flow.σ); # copy to CPU
        dat[] = d[inside(d)];  # update Observable
    end

    # Plot ω_θ contours
    ω_θ!(dat,d,sim,center)
    contour!(dat, levels=[-5,5], colormap=:balance)
    fig
end

# Loop in time
for _ in 1:100
    sim_step!(sim,sim_time(sim)+0.05,remeasure=false)
    ω_θ!(dat,d,sim,center)
end

# using BenchmarkTools
# @btime sim_step!($sim,sim_time($sim)+0.05,remeasure=false); # 64.772 ms (103213 allocations: 5.21 MiB)
# @btime ω_θ!($dat,$d,$sim,$center); # 5.148 ms (2695 allocations: 2.11 MiB) W00T!
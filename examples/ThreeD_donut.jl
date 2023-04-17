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

    # CPU arrays for u and viz
    u = sim.flow.u |> Array
    dat = d[inside(d)] |> Observable;

    # Compute θ-component of ω on CPU
    function ω_θ_data!(dat,d,u,sim,center)
        copyto!(u,sim.flow.u) # copy to CPU container
        dt = sim.L/sim.U
        @inside d[I] = WaterLily.ω_θ(I,SA[1,0,0],center,u)*dt
        dat[] = d[inside(d)]  # update Observable
    end
    ω_θ_data!(dat,d,u,sim,center)

    # Plot flow contours
    contour!(dat, levels=[-5,5], colormap=:balance)
    fig
end

# Loop in time
for _ in 1:100
    sim_step!(sim,sim_time(sim)+0.05) # Runs on GPU
    ω_θ_data!(dat,d,u,sim,center)     # Runs on CPU
end

# using BenchmarkTools
# @btime sim_step!($sim,sim_time($sim)+0.05); # 64.772 ms (103213 allocations: 5.21 MiB)
# @btime ω_θ_data!($dat,$d,$u,$sim,$center); # 45.077 ms (5769052 allocations: 154.08 MiB) Too slow!

# @btime copyto!($u,$sim.flow.u); # 803.700 μs (1 allocation: 16 bytes)
# # @btime copyto!($d,$sim.flow.σ);  # 3x faster (not surprising)
# dt = sim.L/sim.U
# kern(d,center,u,dt) = @inside d[I] = WaterLily.ω_θ(I,SA[1,0,0],center,u)*dt
# @btime kern($d,$center,$u,$dt); # 39.492 ms (5766738 allocations: 152.00 MiB) !!! This is terrible !!!
# @btime $dat[] = $d[inside($d)]; # 2.127 ms (2313 allocations: 2.08 MiB)

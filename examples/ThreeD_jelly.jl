using WaterLily
using StaticArrays
function jelly(p=5;Re=5e2,mem=Array,U=1)
    # Define simulation size, geometry dimensions, & viscosity
    n = 2^p; R = 2n/3; h = 4n-2R; ν = U*R/Re

    # Motion functions
    ω = 2U/R
    @fastmath @inline A(t) = 1 .- SA[1,1,0]*0.1*cos(ω*t)
    @fastmath @inline B(t) = SA[0,0,1]*((cos(ω*t)-1)*R/4-h)
    @fastmath @inline C(t) = SA[0,0,1]*sin(ω*t)*R/4

    # Build jelly from a mapped sphere and plane
    sphere = AutoBody((x,t)->abs(√sum(abs2,x)-R)-1, # sdf
                      (x,t)->A(t).*x+B(t)+C(t))     # map
    plane = AutoBody((x,t)->x[3]-h,(x,t)->x+C(t))
    body =  sphere-plane

    # Initialize simulation and return center for flow viz
    Simulation((n,n,4n),(0,0,-U),R;ν,body,mem,T=Float32)
end

function geom!(geom,md,d,sim,t=WaterLily.time(sim))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,t)
    copyto!(d,sim.flow.σ)      # copy to CPU
    mirrorto!(md,d[inside(d)]) # mirror quadrant
    geom[] = md;               # update Observable
end

function ω!(ω,md,d,sim)
    dt = sim.L/sim.U
    @inside sim.flow.σ[I] = WaterLily.ω_mag(I,sim.flow.u)*dt
    copyto!(d,sim.flow.σ)      # copy to CPU
    mirrorto!(md,d[inside(d)]) # mirror quadrant
    ω[] = md;                  # update Observable
end

function mirrorto!(a,b)
    n = size(b,1)
    a[reverse(1:n),reverse(1:n),:].=b
    a[reverse(n+1:2n),1:n,:].=a[1:n,1:n,:]
    a[:,reverse(n+1:2n),:].=a[:,1:n,:]
    return
end

using CUDA: CUDA
using GLMakie
begin
    # Define geometry and motion on GPU
    sim = jelly(mem=CUDA.CuArray);

    # Create CPU buffer arrays for geometry flow viz 
    d = sim.flow.σ |> Array               # one quadrant
    md = zeros((2,2,1).*size(inside(d)))  # hold mirrored data

    # Set up geometry viz
    geom = md |> Observable;
    geom!(geom,md,d,sim);
    fig, ax, _ = contour(geom, levels=[-1], alpha=0.01,
            axis = (; type = Axis3, aspect = :data))
    hidedecorations!(ax, grid = false)

    #Set up flow viz
    ω = md |> Observable;
    ω!(ω,md,d,sim);
    volume!(ω, algorithm=:mip, colormap=:algae, colorrange=(1,10))
    fig
end

# Loop in time
# record(fig,"jelly.mp4",1:200) do frame
foreach(1:100) do frame
    @show frame
    sim_step!(sim,sim_time(sim)+0.05);
    geom!(geom,md,d,sim);
    ω!(ω,md,d,sim);
    ax.azimuth[] = pi + 0.3sin(2pi * frame / 120)
end
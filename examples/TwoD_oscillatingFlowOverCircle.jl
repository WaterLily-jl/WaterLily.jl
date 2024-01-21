using WaterLily
using StaticArrays
function circle(n,m;Re=250,U=1)
    # define a circle at the domain center
    radius = m/8
    body = AutoBody((x,t)->√sum(abs2, x .- SA[n/2,m/2]) - radius)

    # define time-varying body force and periodic direction
    accelScale = U^2/radius
    timeScale = radius/U
    Simulation((n,m), (U,0), radius; ν=U*radius/Re,g=(i,t)-> i==1 ? 2accelScale*sin(2π/4*t/timeScale) : 0, Δt=0.01, body, perdir=(1,))
end

include("TwoD_plots.jl")
sim_gif!(circle(3*2^6,2^7),duration=20,step=0.05,clims=(-25,25),plotbody=true,levels=20)
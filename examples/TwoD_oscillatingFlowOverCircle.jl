using WaterLily
using StaticArrays
function circle(n,m;Re=250,U=1)
    radius = m/8
    body = AutoBody((x,t)->√sum(abs2, x .- SA[n/2,m/2]) - radius)
    Simulation((n,m), (1.0U,0.), radius; ν=U*radius/Re,g=(i,t)-> i==1 ? 2U^2/radius*sin(2π*t*U/radius/4) : 0, Δt=0.01, body, perdir=(1,))
end

include("TwoD_plots.jl")
sim_gif!(circle(3*2^6,2^7),duration=20,step=0.05,clims=(-10,10),plotbody=true,levels=100)
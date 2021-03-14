using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function circle(n,m;Re=250)
    # Set physical parameters
    U,R,center = 1., m/8., [m/2,m/2]
    ν=U*R/Re
    @show R,ν
    body = AutoBody((x,t)->norm2(x .- center) - R)
    Simulation((n+2,m+2), [U,0.], R; ν, body)
end

circ = circle(3*2^6,2^7);
sim_gif!(circ;duration=0.1)
sim_gif!(circ;duration=10,step=0.25)

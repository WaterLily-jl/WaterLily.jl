using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function circle(radius=8;Re=250,n=10,m=6)
    center, ν = radius*m/2, radius/Re
    body = AutoBody((x,t)->norm2(x .- center) - radius)
    Simulation((n*radius+2,m*radius+2), [1.,0.], radius; ν, body)
end

# sim_gif!(circle(20);duration=10,step=0.25)

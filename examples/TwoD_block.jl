using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function block(L=2^5;Re=250,U=0.25)
    # Set viscosity
    ν=U*L/Re
    @show L,ν

    # Create dynamic block geometry
    function sdf(x,t)
        x[2] -= clamp(x[2],-L/2,L/2)
        norm2(x)-1
    end
    map(x,t) = x.-[3L+L*sin(t*U/L),2L]
    body=AutoBody(sdf,map)

    Simulation((6L+2,4L+2),zeros(2),L;U,ν,body)
end
# sim_gif!(block();duration=4π,step=π/16,remeasure=true)

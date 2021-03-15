using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function block(m;Re=250)
    # Set physical parameters
    U,L,center = 1., m/4., [m/2+0.5,m/2]
    ν=U*L/Re
    @show L,ν

    # block geometry
    body = AutoBody() do x,t
        x .-= center
        x[2] -= clamp(x[2],-L/2,L/2)
        norm2(x)-1
    end

    Simulation((2m+2,m+2),[U,0.],L;body,ν)
end
sim_gif!(block(2^7);duration=1,step=0.25)

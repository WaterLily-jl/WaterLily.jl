using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function block(L=2^5;Re=250,U=1,amp=0,ϵ=0.5,thk=2ϵ+√2)
    # Set viscosity
    ν=U*L/Re
    @show L,ν

    # Create dynamic block geometry
    sdf(x,t) = (x[2] -= clamp(x[2],-L/2,L/2); norm2(x)-thk/2)
    function map(x,t)
        α = amp*cos(t*U/L)
        [cos(α) sin(α); -sin(α) cos(α)] * (x.-[3L+L*sin(t*U/L)+0.01,4L])
    end
    body = AutoBody(sdf,map)

    Simulation((6L+2,6L+2),zeros(2),L;U,ν,body,ϵ)
end
# sim_gif!(block();duration=4π,step=π/16,remeasure=true)
# sim_gif!(block(amp=π/4);duration=8π,step=π/16,remeasure=true,μbody=true,cfill=:Blues,legend=false,border=:none)

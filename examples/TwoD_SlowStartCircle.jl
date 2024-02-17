using WaterLily
using CUDA
function circle(m,n;a0=0.5,Re=250,U=1,f=Array)
    # define a circle at the domain center
    radius = n/8
    body = AutoBody((x,t)->√sum(abs2, x .- (m/4,n/2)) - radius)
    # define time-varying velocity boundary conditions
    Ut(i,t::T) where T = i==1 ? convert(T,a0*t+(1.0+tanh(31.4*(t-1.0/a0)))/2.0*(1-a0*t)) : zero(T)
    Simulation((m,n), Ut, radius; U, ν=U*radius/Re, body, mem=f)
end
include("TwoD_plots.jl")
sim_gif!(circle(2*196,196;f=CUDA.CuArray),duration=20,clims=(-8,8),plotbody=true)
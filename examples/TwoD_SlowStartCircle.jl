using WaterLily
function circle(m,n;a0=0.5,Re=250,U=1)
    # define a circle at the domain center
    radius = n/8
    body = AutoBody((x,t)->√sum(abs2, x .- (m/4,n/2)) - radius)

    # define time-varying body force `g`
    g(i,t) = i==1 ? a0.*(1.0.-tanh.(31.4.*(t.-1.0./a0)))/2. : 0
    Simulation((m,n), (0,0), radius; U, ν=U*radius/Re, body, g)
end
include("TwoD_plots.jl")
sim = circle(2*196,196)
sim_gif!(sim,duration=100,clims=(-8,8),plotbody=true)
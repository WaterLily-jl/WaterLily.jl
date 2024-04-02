using WaterLily
function circle(n,m;κ=1.5,Re=250,U=1)
    # define a circle at the domain center
    radius = m/8
    body = AutoBody((x,t)->√sum(abs2, x .- (n/2,m/2)) - radius)

    # define time-varying body force `g` and periodic direction `perdir`
    accelScale, timeScale = U^2/2radius, κ*radius/U
    g(i,t) = i==1 ? -2accelScale*sin(t/timeScale) : 0 
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, g, perdir=(1,))
end
include("TwoD_plots.jl")
sim_gif!(circle(196,196),duration=20,clims=(-8,8),plotbody=true)
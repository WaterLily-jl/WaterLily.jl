using WaterLily
function circle(radius=8;Re=250,n=(10,6),U=1)
    center = radius*n[2]/2
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation(radius .* n, (U,0), radius; ν=U*radius/Re, body)
end

include("TwoD_plots.jl")
sim_gif!(circle(20),duration=10,clims=(-5,5.05),plotbody=true)

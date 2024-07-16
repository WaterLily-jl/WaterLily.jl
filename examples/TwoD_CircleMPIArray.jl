using WaterLily
# using MPI

# circle simulation
function circle(n,m;Re=250,U=1)
    radius, center = m/8, m/2
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, mem=MPIArray)
end

# test on sim
include("TwoD_plots.jl")
sim = circle(3*2^6,2^7)
sim_gif!(sim,duration=10,clims=(-5,5),plotbody=true)
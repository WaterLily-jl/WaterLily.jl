using WaterLily
using StaticArrays

function circle(n,m;Re=550,U=1,mem=Array,T=Float32)
    radius, center = m/16, m/2+1
    # first circle
    body1 = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    # second circle 1.5 units to the right
    body2 = AutoBody((x,t)->√sum(abs2, x .- SA[1.5center,center]) - radius)
    body = Bodies([body1,body2], [+]) # I want to add them together
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, mem, T)
end
using CUDA


include("TwoD_plots.jl")
sim = circle(3*2^7,2^8,mem=CUDA.CuArray)
sim_gif!(sim,duration=10,clims=(-5,5),plotbody=true)

# # get force on first body
# f1 = WaterLily.∮nds(sim.flow.p,sim.body.bodies[1],WaterLily.time(sim))
# flood(sim.flow.f[:,:,1],clims=(-1,1)) # check that this is only non-zero near the first body

# # force on the second
# f2 = WaterLily.∮nds(sim.flow.p,sim.body.bodies[2],WaterLily.time(sim))
# flood(sim.flow.f[:,:,1],clims=(-1,1)) # check that this is only non-zero near the first body

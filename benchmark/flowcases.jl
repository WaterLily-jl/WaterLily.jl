using WaterLily
using BenchmarkTools
using LinearAlgebra: norm2
using Profile

radius = 8; Re = 250
body = AutoBody((x,t)-> norm2(x .- 2radius) - radius)
gen_sim = ()->Simulation((6radius+2,4radius+2),[1.,0.],radius; body, ν=radius/Re)
sim = gen_sim()

function benchmark_step()
    @benchmark mom_step!($sim.flow,$sim.pois)
end

function benchmark_sim()
    @benchmark sim_step!(sim,10) setup=(sim=$gen_sim())
end

function benchmark_measure(func::Function=measure!)
    flow = Flow((66,66),[1.,0.])
    @benchmark $func($flow,AutoBody(
            (x,t)-> norm2(x .- 2radius) - radius)) setup=(radius=5*(1+rand()))
end

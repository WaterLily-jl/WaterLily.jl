using WaterLily,StaticArrays
include("BioSavart_multilevel.jl")
include("examples/TwoD_plots.jl")

sphere(D,U=1;mem=Array) = Simulation((2D,2D,2D), (U,0,0), D; body=AutoBody((x,t)->√sum(abs2,x .- D)-D/2),ν=U*D/1e4,mem)
sim = sphere(32,mem=Array); ω = ntuple(i->MLArray(sim.flow.σ),3);
# sim = Simulation((2^7,2^6), (1,0), 32; body=AutoBody((x,t)->√sum(abs2,x .- 2^6/2)-32/2),ν=1*32/1e4)
# ω = MLArray(sim.flow.σ);
sim.flow.u⁰ .= sim.flow.u; WaterLily.scale_u!(sim.flow,0)
WaterLily.conv_diff!(sim.flow.f,sim.flow.u⁰,sim.flow.σ,ν=sim.flow.ν)
WaterLily.BDIM!(sim.flow)
BC!(sim.flow.u,sim.flow.U,sim.flow.exitBC)
@gif for i ∈ 1:5
    WaterLily.project!(sim.flow,sim.pois;log=true);
    @show sim.pois.n[end],WaterLily.L₂(sim.pois.levels[1])
    fill_ω!(ω,sim.flow.u);
    biotBC!(sim.flow.u,sim.flow.U,ω);
    pflowBC!(sim.flow.u)
    flood(sim.flow.u[:,:,sim.L,2])
end
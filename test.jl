using WaterLily,StaticArrays

function map(x,t;L,θ)
    s,c = sincos(θ)
    SA[c -s; s c]*(x-SA[L,L])
end
function sdf(ξ,t;L)
    p = ξ-SA[0,clamp(ξ[1],-L/2,L/2)]
    √(p'*p)-2
end
function make_sim(θ;L=32,U=1,Re=100)
    T=typeof(θ)
    body=AutoBody((a,b)->sdf(a,b;L),(a,b)->map(a,b;L,θ))
    Simulation((2L,2L),(U,0),L;ν=U*L/Re,body,T)
end

function step_force!(θ,sim,t=WaterLily.timeNext(sim.flow))
    body = AutoBody((a,b)->sdf(a,b;sim.L),(a,b)->map(a,b;sim.L,θ))
    measure!(sim.flow,body;t,ϵ=sim.ϵ)
    WaterLily.update!(sim.pois)
    mom_step!(sim.flow,sim.pois)
    sum(sim.flow.p)
end
sim = make_sim(π/36);
lift_hist = [step_force!(π/36,sim) for _ ∈ 1:20]
a = step_force!(π/36+1e-4,deepcopy(sim));
b = step_force!(π/36-1e-4,deepcopy(sim));
println("FD grad = ", (a-b)/2e-4)

function step_force(θ,sim⁰)
    sim = make_sim(θ)
    sim.flow.u .= sim⁰.flow.u; sim.flow.p .= sim⁰.flow.p;
    sim.flow.Δt .= WaterLily.time(sim⁰.flow); push!(sim.flow.Δt,sim⁰.flow.Δt[end])
    sim_step!(sim)
    sum(sim.flow.p)
end
using ForwardDiff: derivative
println("AD grad = ", derivative(x->step_force(x,deepcopy(sim)),π/36))
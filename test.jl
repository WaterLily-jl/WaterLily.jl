using WaterLily,StaticArrays

function map(x,t;L,θ)
    s,c = sincos(θ)
    SA[c -s; s c]*(x-SA[L,L])
end
function sdf(ξ,t;L)
    p = ξ-SA[0,clamp(ξ[1],-L/2,L/2)] 
    √(p'*p)-2
end
function make_sim(θ;L=32,U=1,Re=100,T=Float64)
    body=AutoBody((a,b)->sdf(a,b;L),(a,b)->map(a,b;L,θ))
    Simulation((2L,2L),(U,0),L;ν=U*L/Re,body,T)
end
function step_force!(sim)
    sim_step!(sim)
    sum(sim.flow.p)
end
using ForwardDiff: derivative, Dual, Tag
T = Float64 #typeof(Tag(make_sim, Float64))
sim = make_sim(π/36;T);
lift_hist = [step_force!(sim) for _ ∈ 1:20]

function step_force(θ,sim⁰)
    a⁰ = sim⁰.flow
    t = WaterLily.timeNext(a⁰)
    R = inside(a⁰.p)
    body = AutoBody((a,b)->sdf(a,b;sim⁰.L),(a,b)->map(a,b;sim⁰.L,θ))
    T = typeof(WaterLily.sdf(body,WaterLily.loc(0,first(R)),t))
    a = Flow(size(R), a⁰.U; ν=a⁰.ν, T)
    a.u⁰ .= a⁰.u⁰; a.p .= a⁰.p;
    a.Δt .= t-a⁰.Δt[end]; push!(a.Δt,a⁰.Δt[end])
    measure!(a,body;t)
    mom_step!(a,MultiLevelPoisson(a.p,a.μ₀,a.σ))
    sum(a.p)
end
step_force(Dual{Tag{typeof(step_force), Float64}}(π/36,1),sim)
(step_force(π/36+1e-4,sim)-step_force(π/36-1e-4,sim))/2e-4
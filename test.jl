using WaterLily,StaticArrays

function map(x,t;L,θ)
    s,c = sincos(θ)
    SA[c -s; s c]*(x-SA[L,L])
end
function sdf(ξ,t;L)
    p = ξ-SA[0,clamp(ξ[1],-L/2,L/2)] 
    √(p'*p)-2
end
function make_sim(θ;L=32,U=1,Re=100,thk=2+√2,T=Float64)
    body=AutoBody((a,b)->sdf(a,b;L,thk),(a,b)->map(a,b;L,θ))
    Simulation((2L,2L),(U,0),L;ν=U*L/Re,body,ϵ,T)
end
function step_force!(sim)
    sim_step!(sim)
    sum(sim.flow.p)
end
using ForwardDiff: derivative, Dual, Tag
T = Float64 #typeof(Tag(make_sim, Float64))
sim = make_sim(π/36;T);
lift_hist = [step_force!(sim) for _ ∈ 1:20]

function step_force(θ,sim₀)
    @show θ
    t = WaterLily.timeNext(sim₀.flow)
    R = inside(sim₀.flow.p)
    body = AutoBody((a,b)->sdf(a,b;sim₀.L),(a,b)->map(a,b;sim₀.L,θ))
    dᵢ,nᵢ,Vᵢ = measure(body,WaterLily.loc(1,first(R)),t)
    @show dᵢ
    @show nᵢ
    @show Vᵢ  
    # flow = Flow()
    # measure!(a::Flow{N},body::AbstractBody;t)
    # mom_step!(flow,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ))
    # sum(flow.p)
    dᵢ
end
derivative(θ->step_force(θ,sim),π/36)
step_force(Dual{typeof(Tag(step_force, Float64))}(π/36,1),sim)
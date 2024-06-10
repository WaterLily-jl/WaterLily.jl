using WaterLily,StaticArrays

global θ₀=π/36
global h₀=0

function make_sim(;L=32,U=1,Re=100,ϵ=0.5,thk=2ϵ+√2,T=Float64)
    cen = SA[L,L]
    h(t) = h₀
    θ(t) = θ₀
    function map(x,t)
        s,c = sincos(θ(t))
        SA[c -s; s c]*(x-cen-SA[0,h(t)])
    end
    function sdf(ξ,t) # Line segment
        p = ξ-SA[0,clamp(ξ[1],-L/2,L/2)] 
        √(p'*p)-thk/2
    end
    Simulation((2L,2L),(U,0),L;ν=U*L/Re,body=AutoBody(sdf,map),ϵ,T)
end
function step_force!(sim,θ)
    global θ₀ = θ
    sim_step!(sim)
    sum(sim.flow.p)
end
using ForwardDiff: derivative, Dual, Tag
R = Float64
T = typeof(Tag(make_sim, R))
sim = make_sim(;T);
# lift_hist = [step_force!(sim,π/36) for _ ∈ 1:20]
step_force!(sim,Dual{Float64}(π/36,1))
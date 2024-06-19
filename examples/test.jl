using WaterLily,StaticArrays

function make_sim(θ;L=32,U=1,Re=100,mem=Array)
    function map(x,t)
        s,c = sincos(θ[])
        SA[c -s; s c]*(x-SA[L,L])
    end
    function sdf(ξ,t)
        p = ξ-SA[0,clamp(ξ[1],-L/2,L/2)]
        √(p'*p)-2
    end
    Simulation((2L,2L),(U,0),L,ν=U*L/Re,body=AutoBody(sdf,map),T=typeof(θ[]),mem=mem)
end

sim = make_sim(0f0);
a,b = sim.flow,sim.pois;
WaterLily.mom_step!(a,b)
@time WaterLily.mom_step!(a,b) # test allocations

function step_force!(sim)
    sim_step!(sim)
    sum(sim.flow.p)
end

θ₀ = Float32(π/36)
θ = Ref(θ₀) # wrap the parameter in a Ref so it can be updated

sim = make_sim(θ);
lift_hist = [step_force!(sim) for _ ∈ 1:20]
θ[] = θ₀+0.001f0; a = step_force!(deepcopy(sim)); # use a copy to avoid updating sim
θ[] = θ₀-0.001f0; b = step_force!(deepcopy(sim)); # use a copy to avoid updating sim
θ[] = θ₀; c = step_force!(sim);
println("sim value and FD partial = ", (c,(a-b)/0.002f0))

using ForwardDiff: Dual,Tag
T = typeof(Tag(step_force!,typeof(θ₀))) # make a tag
θAD = Ref(Dual{T}(θ₀,0))                # wrap the Dual parameter in a Ref
simAD = make_sim(θAD);                  # make a sim of the correct type
lift_histAD = [step_force!(simAD) for _ ∈ 1:20] # still works
θAD[] = Dual{T}(θ₀,1)                   # update partial to take derivative
println("simAD = ", step_force!(simAD))
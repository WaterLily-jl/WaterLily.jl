using WaterLily,StaticArrays,CUDA

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

function step_force!(sim)
    sim_step!(sim)
    sum(sim.flow.p)
end

mem = Array
θ = Ref(π/36) # wrap the parameter in a Ref so it can be updated

sim = make_sim(θ; mem);
lift_hist = [step_force!(sim) for _ ∈ 1:20]
θ[] = π/36+1e-4; a = step_force!(deepcopy(sim)); # use a copy to avoid updating sim
θ[] = π/36-1e-4; b = step_force!(deepcopy(sim)); # use a copy to avoid updating sim
θ[] = π/36; c = step_force!(sim);
println("sim value and FD partial = ", (c,(a-b)/2e-4))

using ForwardDiff: Dual,Tag
T = typeof(Tag(step_force!, Float64)) # make a tag
θAD = Ref(Dual{T}(π/36, 0.))          # wrap the Dual parameter in a Ref
simAD = make_sim(θAD; mem);          # make a sim of the correct type
lift_histAD = [step_force!(simAD) for _ ∈ 1:20] # still works
θAD[] = Dual{T}(π/36,1.)              # update partial to take derivative
println("simAD = ", step_force!(simAD))
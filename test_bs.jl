using WaterLily,StaticArrays
include("BioSavart_multilevel.jl")
include("examples/TwoD_plots.jl")

function BC_BiotSavart!(u,U,ω)
    fill_ω!(ω,sim.flow.u); 
    N,n = size_u(u)
    for j ∈ 1:n, i ∈ 1:n
        if i==j # Normal direction
            for s ∈ (1,2,N[j])
                @loop u[I,i] = u_ω(i,I,ω)+U[i] over I ∈ slice(N,s,j)
            end
        else    # Tangential directions
            for s ∈ (1,N[j])
                @loop u[I,i] = u_ω(i,I,ω)+U[i] over I ∈ slice(N,s,j)
            end
        end
    end
end


sphere(D,U=1;mem=Array) = Simulation((2D,2D,2D), (U,0,0), D; body=AutoBody((x,t)->√sum(abs2,x .- D)-D/2),ν=U*D/1e4,mem)
sim = sphere(32,mem=Array); ω = ntuple(i->MLArray(sim.flow.σ),3);

# sim = Simulation((2^7,2^6), (1,0), 32; body=AutoBody((x,t)->√sum(abs2,x .- 2^6/2)-32/2),ν=1*32/1e4)
# ω = MLArray(sim.flow.σ);

sim.flow.u⁰ .= sim.flow.u; WaterLily.scale_u!(sim.flow,0)
WaterLily.conv_diff!(sim.flow.f,sim.flow.u⁰,sim.flow.σ,ν=sim.flow.ν)
WaterLily.BDIM!(sim.flow)
BC!(sim.flow.u,sim.flow.U,sim.flow.exitBC)
@gif for i ∈ 1:10
    WaterLily.project!(sim.flow,sim.pois;log=true);
    @show sim.pois.n[end], WaterLily.L₂(sim.pois.levels[1])
    BC_BiotSavart!(sim.flow.u,sim.flow.U,ω);
    # @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
    # flood(sim.flow.σ[:,:,32])
    flood(sim.flow.u[:,:,32,2])
end
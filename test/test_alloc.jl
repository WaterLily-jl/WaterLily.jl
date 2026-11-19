using BenchmarkTools, Printf

backend != "SIMD" && throw(ArgumentError("KernelAbstractions backend not allowed to run allocations tests, use SIMD backend"))

# Run benchmark `b`, print the bytes allocated on its best (minimum) run next to the test `case`, and test that it is zero
function noalloc(case, b)
    tune!(b); r = run(b)
    @printf("▶ %-34s %d B\n", case, r.memory)
    @test r.memory == 0
end

@testset "mom_step! allocations" begin
    function Sim(θ;L=32,U=1,Re=100,perdir=(),λ=quick)
        function map(x,t)
            s,c = sincos(θ)
            SA[c -s; s c]*(x-SA[L,L])
        end
        function sdf(ξ,t)
            p = ξ-SA[0,clamp(ξ[1],-L/2,L/2)]
            √(p'*p)-2
        end
        Simulation((20L,20L),(U,0),L,ν=U*L/Re,body=AutoBody(sdf,map),T=Float32,perdir=perdir,λ=λ)
    end
    sim = Sim(Float32(π/36)) # default λ=quick
    sim_step!(sim)
    noalloc("mom_step! (λ=quick)", @benchmarkable mom_step!($sim.flow, $sim.pois) samples=100)

    sim_cds = Sim(Float32(π/36); λ=cds) # convective scheme selected at construction; mom_step! must still specialize
    sim_step!(sim_cds)
    noalloc("mom_step! (λ=cds)", @benchmarkable mom_step!($sim_cds.flow, $sim_cds.pois) samples=100)

    sim = Sim(Float32(π/36); perdir=(2,)) # periodic conv_diff! fluxes must not allocate either
    sim_step!(sim)
    noalloc("mom_step! (perdir=(2,))", @benchmarkable mom_step!($sim.flow, $sim.pois) samples=100)
    # concretely typed Simulation and Poisson fields: no runtime dispatch
    noalloc("measure! (perdir=(2,))", @benchmarkable measure!($sim) samples=100)
end

@testset "semi-coarsening + remeasure allocations" begin
    # Non-square (4:1) domain + moving body: exercises semi-coarsening and the per-step remeasure
    # path through the multigrid V-cycle, where a boxed V-cycle call would allocate ~2 KiB.
    function SimAniso(θ; L=32, U=1, Re=100)
        function map(x,t)
            s,c = sincos(θ + t*U/L)
            SA[c -s; s c]*(x-SA[2L,L])
        end
        function sdf(ξ,t)
            p = ξ-SA[0,clamp(ξ[1],-L,L)]
            √(p'*p)-2
        end
        Simulation((20L,5L),(U,0),L,ν=U*L/Re,body=AutoBody(sdf,map),T=Float32)
    end
    sim = SimAniso(Float32(π/36))
    sim_step!(sim)
    noalloc("sim_step! (aniso, remeasure=true)", @benchmarkable sim_step!($sim; remeasure=true) samples=100) # full step incl. remeasure
end

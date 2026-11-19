using BenchmarkTools, Printf

backend != "SIMD" && throw(ArgumentError("KernelAbstractions backend not allowed to run allocations tests, use SIMD backend"))

# Print the memory allocated on the best (minimum) run of a benchmark, labelled with the test case it belongs to
report(case, bytes) = @printf("▶ %-34s %6.2f KiB\n", case, bytes/1024)

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
    b = @benchmarkable mom_step!($sim.flow, $sim.pois) samples=100; tune!(b) # check 100 times
    r = run(b)
    report("mom_step! (λ=quick)", r.memory)
    @test r.memory == 0 # no allocations on the best run (commit 7458970: 0 B)

    sim_cds = Sim(Float32(π/36); λ=cds) # convective scheme selected at construction; mom_step! must still specialize
    sim_step!(sim_cds)
    b = @benchmarkable mom_step!($sim_cds.flow, $sim_cds.pois) samples=100; tune!(b) # check 100 times
    r = run(b)
    report("mom_step! (λ=cds)", r.memory)
    @test r.memory == 0 # no allocations on the best run (commit 7458970: 0 B)

    sim = Sim(Float32(π/36); perdir=(2,)) # periodic conv_diff! fluxes must not allocate either
    sim_step!(sim)
    b = @benchmarkable mom_step!($sim.flow, $sim.pois) samples=100; tune!(b) # check 100 times
    r = run(b)
    report("mom_step! (perdir=(2,))", r.memory)
    @test r.memory == 0 # no allocations on the best run (commit 7458970: 0 B)

    b = @benchmarkable measure!($sim) samples=100; tune!(b) # check 100 times
    r = run(b)
    report("measure! (perdir=(2,))", r.memory)
    @test r.memory < 1024 # less than 1 KiB on the best run (commit 7458970: 0.44 KiB)
end

@testset "semi-coarsening + remeasure allocations" begin
    # Non-square (4:1) domain + moving body: exercises semi-coarsening and the per-step remeasure
    # path through the multigrid V-cycle. Tight bound to catch per-V-cycle boxing (commit 7458970: 0.22 KiB).
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
    b = @benchmarkable sim_step!($sim; remeasure=true) samples=100; tune!(b) # full step incl. remeasure
    r = run(b)
    report("sim_step! (aniso, remeasure=true)", r.memory)
    @test r.memory < 512 # less than 0.5 KiB; 0.22 KiB at commit 7458970, ~2 KiB if a V-cycle call boxes
end

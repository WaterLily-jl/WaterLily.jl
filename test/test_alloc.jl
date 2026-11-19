using BenchmarkTools, Printf

backend != "SIMD" && throw(ArgumentError("KernelAbstractions backend not allowed to run allocations tests, use SIMD backend"))
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
    println("▶ Allocated "*@sprintf("%.0f", r.memory/1e3)*" KiB")
    @test r.memory < 50000 # less than 50 KiB allocated on the best mom_step! run (commit f721343 ≈ 8 KiB)

    sim_cds = Sim(Float32(π/36); λ=cds) # convective scheme selected at construction; mom_step! must still specialize
    sim_step!(sim_cds)
    b = @benchmarkable mom_step!($sim_cds.flow, $sim_cds.pois) samples=100; tune!(b) # check 100 times
    r = run(b)
    println("▶ Allocated "*@sprintf("%.0f", r.memory/1e3)*" KiB")
    @test r.memory < 50000 # less than 50 KiB allocated on the best mom_step! run (commit f721343 ≈ 8 KiB)

    sim = Sim(Float32(π/36); perdir=(2,))
    sim_step!(sim)
    b = @benchmarkable mom_step!($sim.flow, $sim.pois) samples=100; tune!(b) # check 100 times
    r = run(b)
    println("▶ Allocated "*@sprintf("%.0f", r.memory/1e3)*" KiB")
    @test r.memory < 50000 # less than 50 KiB allocated on the best mom_step! run (commit f721343 ≈ 8 KiB)

    b = @benchmarkable measure!($sim) samples=100; tune!(b) # check 100 times
    r = run(b)
    println("▶ Allocated "*@sprintf("%.0f", r.memory/1e3)*" KiB")
    @test r.memory < 50000 # less than 50 KiB allocated on the best mom_step! run (commit f721343 ≈ 8 KiB)

    # Verify AbstractFlow type widening did not introduce allocations via dynamic dispatch
    # flow_ctor produces a Flow <: AbstractFlow; mom_step! must still be fully compiled
    let θ=Float32(π/36), L=32, U=1, Re=100
        function map2(x,t); s,c=sincos(θ); SA[c -s;s c]*(x-SA[L,L]) end
        function sdf2(ξ,t); p=ξ-SA[0,clamp(ξ[1],-L/2,L/2)]; √(p'*p)-2 end
        sim2 = Simulation((20L,20L),(U,0),L;ν=U*L/Re,body=AutoBody(sdf2,map2),T=Float32,
                          flow_ctor=(d,u;kw...)->Flow(d,u;kw...))
        sim_step!(sim2)
        b = @benchmarkable mom_step!($sim2.flow, $sim2.pois) samples=100; tune!(b)
        r = run(b)
        println("▶ Allocated (flow_ctor path) "*@sprintf("%.0f", r.memory/1e3)*" KiB")
        @test r.memory < 50000 # AbstractFlow dispatch must not allocate more than concrete Flow
    end
end

@testset "semi-coarsening + remeasure allocations" begin
    # Non-square (4:1) domain + moving body: exercises semi-coarsening and the per-step remeasure
    # path through the multigrid V-cycle. Tight bound to catch per-V-cycle boxing (clean ≈ 0.45 KiB).
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
    println("▶ Allocated (aniso sim_step! remeasure) "*@sprintf("%.2f", r.memory/1e3)*" KiB")
    @test r.memory < 1000 # less than 1 KiB; ~0.45 KiB clean, ~2 KiB if a V-cycle call boxes
end

using BenchmarkTools, Printf

backend == "KernelAbstractions" && (set_backend("SIMD"); exit())

@testset "mom_step! allocations" begin
    function Sim(θ;L=32,U=1,Re=100,perdir=())
        function map(x,t)
            s,c = sincos(θ)
            SA[c -s; s c]*(x-SA[L,L])
        end
        function sdf(ξ,t)
            p = ξ-SA[0,clamp(ξ[1],-L/2,L/2)]
            √(p'*p)-2
        end
        Simulation((20L,20L),(U,0),L,ν=U*L/Re,body=AutoBody(sdf,map),T=Float32,perdir=perdir)
    end
    sim = Sim(Float32(π/36))
    sim_step!(sim)
    b = @benchmarkable mom_step!($sim.flow, $sim.pois) samples=100; tune!(b) # check 100 times
    r = run(b)
    println("▶ Allocated "*@sprintf("%.0f", r.memory/1e3)*" KiB")
    @test r.memory < 50000 # less than 50 KiB allocated on the best mom_step! run (commit f721343 ≈ 8 KiB)

    sim = Sim(Float32(π/36); perdir=(2,))
    sim_step!(sim)
    b = @benchmarkable mom_step!($sim.flow, $sim.pois) samples=100; tune!(b) # check 100 times
    r = run(b)
    println("▶ Allocated "*@sprintf("%.0f", r.memory/1e3)*" KiB")
    @test r.memory < 50000 # less than 50 KiB allocated on the best mom_step! run (commit f721343 ≈ 8 KiB)
end
@testset "WaterLily.jl" begin
    radius = 8; ν=radius/250; T=Float32; nm = radius.*(4,4)
    circle(x,t) = √sum(abs2,x .- 2radius) - radius
    move(x,t) = x-SA[t,0]
    accel(x,t) = x-SA[2t^2,0]
    plate(x,t) = √sum(abs2,x - SA[clamp(x[1],-radius+2,radius-2),0])-2
    function rotate(x,t)
        s,c = sincos(t/radius+1); R = SA[c s ; -s c]
        R * (x .- 2radius)
    end
    function bend(xy,t) # into ≈ circular arc
        x,y = xy .- 2radius; κ = 2t/radius^2+0.2f0/radius
        return SA[x+x^3*κ^2/6,y-x^2*κ/2]
    end
    # Test sim_time, and sim_step! stopping time
    sim = Simulation(nm,(1,0),radius; body=AutoBody(circle), ν, T)
    @test sim_time(sim) == 0
    sim_step!(sim,0.1,remeasure=false)
    @test sim_time(sim) ≥ 0.1 > sum(sim.flow.Δt[1:end-2])*sim.U/sim.L
    for mem ∈ arrays, exitBC ∈ (true,false)
        # Test that remeasure works perfectly when V = U = 1
        sim = Simulation(nm,(1,0),radius; body=AutoBody(circle,move), ν, T, mem, exitBC)
        sim_step!(sim)
        @test all(sim.flow.u[:,radius,1].≈1)
        # @test all(sim.pois.n .== 0)
        # Test accelerating from U=0 to U=1
        sim = Simulation(nm,(0,0),radius; U=1, body=AutoBody(circle,accel), ν, T, mem, exitBC)
        sim_step!(sim)
        @test length(sim.pois.n)==2 && all(sim.pois.n .<5)
        @test maximum(sim.flow.u) > maximum(sim.flow.V) > 0
        # Test that non-uniform V doesn't break
        sim = Simulation(nm,(0,0),radius; U=1, body=AutoBody(plate,rotate), ν, T, mem, exitBC)
        sim_step!(sim)
        @test length(sim.pois.n)==2 && all(sim.pois.n .<5)
        @test 1 > sim.flow.Δt[end] > 0.5
        # Test that divergent V doesn't break
        sim = Simulation(nm,(0,0),radius; U=1, body=AutoBody(plate,bend), ν, T, mem, exitBC)
        sim_step!(sim)
        @test length(sim.pois.n)==2 && all(sim.pois.n .<5)
        @test 1.2 > sim.flow.Δt[end] > 0.8
    end
    # Test flow_ctor factory: explicit lambda wrapping Flow produces a working simulation
    sim = Simulation(nm,(1,0),radius; body=AutoBody(circle), ν, T,
                     flow_ctor=(d,u;kw...)->Flow(d,u;kw...))
    @test sim.flow isa Flow
    sim_step!(sim,0.5,remeasure=false)
    @test all(isfinite, sim.flow.u)

    # Test pois_ctor factory: explicit lambda wrapping MultiLevelPoisson produces a working simulation
    sim = Simulation(nm,(1,0),radius; body=AutoBody(circle), ν, T,
                     pois_ctor=flow->MultiLevelPoisson(flow.p,flow.μ₀,flow.σ))
    @test sim.pois isa MultiLevelPoisson
    sim_step!(sim,0.5,remeasure=false)
    @test all(isfinite, sim.flow.u)

    # Initial-condition keyword `u0` and its deprecated alias `uλ`
    # Test to be removed in WL 2.0
    ic(i,x) = i==1 ? 2.0f0 : 0.0f0
    zeroic(i,x) = 0.0f0
    # `u0` sets the interior initial velocity
    simu0 = Simulation((16,16),(1,0),16; u0=ic, T=Float32)
    @test all(≈(2.0f0), simu0.flow.u[3:14,3:14,1])
    # `uλ` is a deprecated alias for `u0`: it warns once and gives the same field
    simuλ = @test_logs (:warn, r"uλ.*deprecated") match_mode=:any Simulation((16,16),(1,0),16; uλ=ic, T=Float32)
    @test simuλ.flow.u == simu0.flow.u
    # `u0` takes precedence over `uλ` when both are supplied
    simboth = Simulation((16,16),(1,0),16; u0=ic, uλ=zeroic, T=Float32)
    @test simboth.flow.u == simu0.flow.u
    # the resolver returns the active initial condition
    @test WaterLily.ic_kwarg(ic, nothing) === ic
    @test WaterLily.ic_kwarg(nothing, nothing) === nothing
end

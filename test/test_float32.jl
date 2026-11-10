@testset "Float32 simulation on every backend (Metal has no Float64)" begin
    # Same Float32 case on the CPU and on every requested backend: fields and integrals must agree.
    # Integrals are accumulated in the field type, so Metal (no Float64) runs the same code.
    N = 64
    function circle_sim(mem; T=Float32)
        L = T(N / 4); c = T(N / 2)
        body = AutoBody((x, t) -> √sum(abs2, x .- c) - L / 2)
        return Simulation((N, N), (one(T), zero(T)), L; ν=L / T(200), body, T, mem, exitBC=true)
    end
    tstep(sim) = sim_step!(sim, oftype(sim.flow.Δt[end], 1); remeasure=false)
    x₀ = SVector{2,Float32}(N / 2, N / 2) # circle centre
    x₀_FP64 = SVector{2,Float64}(x₀)      # converted to the field type before it reaches the kernel

    ref = circle_sim(Array); tstep(ref)
    Fp_ref = WaterLily.pressure_force(ref); Fv_ref = WaterLily.viscous_force(ref)
    M_ref = WaterLily.total_moment(x₀, ref)
    @test Fp_ref isa Vector{Float32} && Fv_ref isa Vector{Float32} && M_ref isa Vector{Float32}
    @test abs(Fp_ref[1]) > 0                     # drag
    @test abs(Fp_ref[2]) < 1e-2 * abs(Fp_ref[1]) # symmetric body: no lift

    for f ∈ arrays
        sim = circle_sim(f); tstep(sim)
        @test eltype(sim.flow.u) == Float32 && eltype(sim.flow.p) == Float32 && eltype(sim.flow.μ₀) == Float32
        @test length(sim.flow.Δt) == length(ref.flow.Δt) && sim.flow.Δt ≈ ref.flow.Δt
        @test maximum(abs, Array(sim.flow.u) .- ref.flow.u) < 1f-4
        @test maximum(abs, Array(sim.flow.p) .- ref.flow.p) < 2f-3 # multigrid: converged to solver tolerance only
        @test maximum(abs, Array(sim.flow.μ₀) .- ref.flow.μ₀) < 1f-6
        Fp = WaterLily.pressure_force(sim); Fv = WaterLily.viscous_force(sim)
        @test Fp isa Vector{Float32} && Fv isa Vector{Float32}
        @test Fp ≈ Fp_ref rtol=1e-4
        @test Fv ≈ Fv_ref rtol=1e-4
        @test WaterLily.total_force(sim) ≈ Fp_ref .+ Fv_ref rtol=1e-4
        @test WaterLily.total_moment(x₀, sim) ≈ M_ref atol=1e-5
        @test WaterLily.total_moment(x₀_FP64, sim) == WaterLily.total_moment(x₀, sim)
    end
end

@testset "Float32 kernels stay Float32 (GPU backends without Float64 support)" begin
    # A Float32 simulation with an immersed body must not promote to Float64 inside the kernels:
    # Apple Metal has no Float64, so any promotion makes the kernel fail to compile. This test
    # runs the same case on the CPU and on every requested array backend and checks that the
    # fields and the force integrals agree, and that forces are still accumulated in Float64.
    function circle_sim(mem; T=Float32, N=64)
        L = T(N / 4); c = T(N / 2)
        body = AutoBody((x, t) -> √sum(abs2, x .- c) - L / 2)
        return Simulation((N, N), (one(T), zero(T)), L; ν=L / T(200), body, T, mem, exitBC=true)
    end
    tstep(sim) = sim_step!(sim, oftype(sim.flow.Δt[end], 1); remeasure=false)

    ref = circle_sim(Array); tstep(ref)
    Fp_ref = WaterLily.pressure_force(ref); Fv_ref = WaterLily.viscous_force(ref)
    @test eltype(Fp_ref) == Float64 && eltype(Fv_ref) == Float64   # accumulation stays Float64
    @test abs(Fp_ref[1]) > 0                                        # a drag force exists
    @test abs(Fp_ref[2]) < 1e-2 * abs(Fp_ref[1])                    # symmetric body: no lift

    for f ∈ arrays
        sim = circle_sim(f); tstep(sim)                            # compiles and runs on this backend
        @test eltype(sim.flow.u) == Float32 && eltype(sim.flow.p) == Float32 && eltype(sim.flow.μ₀) == Float32
        @test sim.flow.Δt ≈ ref.flow.Δt                           # identical time stepping
        @test maximum(abs, Array(sim.flow.u) .- ref.flow.u) < 1f-4   # identical velocity field
        # pressure comes from an iterative multigrid solve whose reduction order differs per
        # backend, so it is only converged to the solver tolerance: compare cell-wise
        @test maximum(abs, Array(sim.flow.p) .- ref.flow.p) < 2f-3
        @test maximum(abs, Array(sim.flow.μ₀) .- ref.flow.μ₀) < 1f-6 # identical kernel moments from measure!
        Fp = WaterLily.pressure_force(sim); Fv = WaterLily.viscous_force(sim)
        @test eltype(Fp) == Float64 && eltype(Fv) == Float64
        @test Fp ≈ Fp_ref rtol=1e-4                                # identical force integrals
        @test Fv ≈ Fv_ref rtol=1e-4
        # the reductions must be pure host code: the result is a plain Vector on every backend
        @test Fp isa Vector{Float64} && Fv isa Vector{Float64}
    end
end

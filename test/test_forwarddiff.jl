@testset "ForwardDiff" begin
    using ForwardDiff
    # Bypass extract_jacobian/valtype in ForwardDiff.jl so nested FD
    # works inside GPU kernels. Sanity-check that they match stock ForwardDiff,
    # both with a plain Float input and with an outer-Dual eltype (the nested
    # case that actually crashed extract_jacobian on GPU codegen).
    sdfn(ξ) = √sum(abs2, ξ) - 1
    rotmap(x, θ) = SA[cos(θ) -sin(θ); sin(θ) cos(θ)] * x
    x0 = SVector(0.5, 0.7); θ0 = 0.3
    @test WaterLily.gradient(sdfn, x0) ≈ ForwardDiff.gradient(sdfn, x0)
    @test WaterLily.jacobian(y -> rotmap(y, θ0), x0) ≈ ForwardDiff.jacobian(y -> rotmap(y, θ0), x0)
    @test WaterLily.derivative(t -> rotmap(x0, t), θ0) ≈ ForwardDiff.derivative(t -> rotmap(x0, t), θ0)
    let outer_tag = typeof(ForwardDiff.Tag(identity, Float64)),
        θd = ForwardDiff.Dual{outer_tag}(θ0, 1.0),
        ref = ForwardDiff.derivative(t -> sum(ForwardDiff.jacobian(y -> rotmap(y, t), x0)), θ0)
        @test ForwardDiff.partials(sum(WaterLily.jacobian(y -> rotmap(y, θd), x0)), 1) ≈ ref
            end

    # Tight kernel-level reproducer of the original GPU bug: call AutoBody.measure
    # inside a @kernel under outer Dual eltype. Without the fix, AutoBody.measure
    # uses stock ForwardDiff.jacobian → extract_jacobian → valtype → DualMismatchError
    # inside the kernel and crashes with KernelException. The body is a line segment
    # (not rotationally symmetric), so the integrated normal genuinely depends on θ.
    function measure_sum(θ, mem; L=16)
        body = AutoBody((ξ, _) -> √sum(abs2, ξ - SA[0, clamp(ξ[1], -L/2, L/2)]) - 2,
                        (x, _) -> SA[cos(θ) -sin(θ); sin(θ) cos(θ)] * (x - SA[L, L]))
        out = mem(zeros(typeof(θ), 2L, 2L))
        WaterLily.@loop out[I] = WaterLily.measure(body, WaterLily.loc(0, I, typeof(θ)), zero(typeof(θ)))[2][1] over I ∈ CartesianIndices(out)
        sum(out)
    end
    cpu_kd = ForwardDiff.derivative(t -> measure_sum(t, Array), 0.3)
    for f ∈ arrays
        @test ForwardDiff.derivative(t -> measure_sum(t, f), 0.3) ≈ cpu_kd rtol=1e-3
    end

    # A RigidMap whose position, velocity AND time carry an outer Dual, as in a differentiated
    # flow, where Δt depends on the parameter. The velocity overload used to fire on any Dual
    # time, mistaking the outer one for `derivative`'s seed: -Inf/NaN on the CPU and a
    # MethodError inside a kernel.
    function rigid_measure(p)
        body = AutoBody((x,_)->√sum(abs2,x)-4, RigidMap(SA[16,16.8+p]; V=SA[zero(p),0.3+2p], ω=p))
        d,n,V = measure(body, SA[14.0,15.0+zero(p)], 0.5+3p)
        [d; n; V]
    end
    h = 1e-6
    @test ForwardDiff.derivative(rigid_measure, 0.0) ≈ (rigid_measure(h)-rigid_measure(-h))/2h rtol=1e-6
    function rigid_sum(p, mem; L=16)
        body = AutoBody((x,_)->√sum(abs2,x)-4, RigidMap(SA[L,L+p]; V=SA[zero(p),0.3+2p], ω=p))
        out = mem(zeros(typeof(p), 2L, 2L)); t = 0.5+3p
        WaterLily.@loop out[I] = sum(WaterLily.measure(body, WaterLily.loc(0, I, typeof(p)), t)[3]) over I ∈ CartesianIndices(out)
        sum(out)
    end
    h = 1e-6
    drigid_fd = (rigid_sum(h, Array) - rigid_sum(-h, Array)) / 2h
    for f ∈ arrays
        @test ForwardDiff.derivative(p -> rigid_sum(p, f), 0.0) ≈ drigid_fd rtol=1e-6
    end

    # End-to-end ForwardDiff AD
    # ∂/∂Re of KE for TGV
    function tgv_sim(Re, mem)
        sim,_ = TGVsim(mem; Re)
        sim_step!(sim,π/100)
        WaterLily.@loop sim.flow.σ[I] = WaterLily.ke(I, sim.flow.u) over I ∈ inside(sim.flow.p)
        sum(@view sim.flow.σ[inside(sim.flow.p)])
    end
    # ∂/∂Lift of spinning cylinder lift generation
    rot(θ) = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]  # rotation matrix
    function spinning(ξ, mem; D=16, Re=500)
        C,R,U = SA[D,D],D÷2,1
        body = AutoBody((x,t)->√(x'*x)-R,          # circle sdf
                        (x,t)->rot(ξ*U*t/R)*(x-C)) # center & spin!
        Simulation((2D,2D), (U,0), D; ν=U*D/Re, body, mem, T=typeof(ξ))
    end
    function spin_sim(ξ, mem)
        sim = spinning(ξ, mem)
        sim_step!(sim, 1; remeasure=false)
        WaterLily.total_force(sim)[2]/(ξ^2*sim.U^2*sim.L)
    end
    # ∂/∂θ of an L²-norm-like pressure scalar for a θ-rotated body (pin_pressure! makes sum(p) ≈ 0)
    function rotating(θ, mem; L=32, U=1, Re=100)
        s, c = sincos(θ)
        body = AutoBody((ξ, _) -> √sum(abs2, ξ - SA[0, clamp(ξ[1], -L/2, L/2)]) - 2,
                        (x, _) -> SA[c -s; s c] * (x - SA[L, L]))
        Simulation((2L, 2L), (U, 0), L; ν=U*L/Re, body, T=typeof(θ), mem)
    end
    rot_sim(θ, mem) = (sim = rotating(θ, mem); sim_step!(sim; max_steps=10); sum(abs2, sim.flow.p))

    # Compare derivative between FD, AD_CPU and AD_GPU
    h = 1
    dtgv_fd = (tgv_sim(1e2+h, Array) - tgv_sim(1e2-h, Array)) / 2h
    h = 1e-6
    dspin_fd = (spin_sim(2+h, Array) - spin_sim(2-h, Array)) / 2h
    h = π/36/100
    drot_fd = (rot_sim(π/36+h, Array) - rot_sim(π/36-h, Array)) / 2h

    for f ∈ arrays
        @test ForwardDiff.derivative(x -> tgv_sim(x, f), 1e2) ≈ dtgv_fd rtol=1e-1
        @test ForwardDiff.derivative(x -> spin_sim(x, f), 2.0) ≈ dspin_fd rtol=√1e-6 # in Float32 the derivative moves by up to 17% when ν moves by one ulp
        @test ForwardDiff.derivative(x -> rot_sim(x, f), π/36) ≈ drot_fd rtol=1e-3
    end
end

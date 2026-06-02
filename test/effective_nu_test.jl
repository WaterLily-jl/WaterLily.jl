# Regression test for the closure-valued `ν` path. The hot loop inside
# `conv_diff!` was changed from `ν*∂(...)` to `_νf(ν,j,I)*∂(...)`. With a
# scalar ν the inlined call compiles to identical code and produces
# identical numerical output. With a constant closure `I->ν_val` the
# result must also be bit-identical (or within a few ULP in Float32).
# The effective viscosity is computed on the fly — no stored array.

using Test
using WaterLily
using StaticArrays

@testset "Closure-valued effective viscosity" begin

    @testset "scalar ν unchanged" begin
        U = (2/3, -1/3)
        N = (16, 16)
        a = WaterLily.Flow(N, U; T=Float32, ν=Float32(0.01))
        @test a.ν isa Float32
        @test a.ν == Float32(0.01)
        Δt = WaterLily.CFL(a)
        @test isfinite(Δt) && Δt > 0
    end

    @testset "constant closure ≈ scalar" begin
        # Two flows, same parameters: one scalar ν, one closure ν(I)=ν_val.
        # After one conv_diff! they must agree to within a few Float32 ULP.
        U = (2/3, -1/3)
        N = (16, 16)
        ν_val = Float32(0.01)

        a_scalar = WaterLily.Flow(N, U; T=Float32, ν=ν_val)
        a_clos   = WaterLily.Flow(N, U; T=Float32, ν=(I -> ν_val))
        @test a_clos.ν isa Function

        for I in CartesianIndices(a_scalar.u)
            v = sinpi((I.I[1] - 1) / N[1]) + cospi((I.I[2] - 1) / N[2])
            a_scalar.u[I] = a_clos.u[I] = Float32(v)
        end
        Φs, Φc = similar(a_scalar.p), similar(a_clos.p)
        WaterLily.conv_diff!(a_scalar.f, a_scalar.u, Φs, WaterLily.quick; ν=a_scalar.ν)
        WaterLily.conv_diff!(a_clos.f,   a_clos.u,   Φc, WaterLily.quick; ν=a_clos.ν)

        err = maximum(abs.(a_scalar.f .- a_clos.f))
        @test err ≤ 5 * eps(Float32) * maximum(abs.(a_scalar.f))
    end

    @testset "closure enables spatially varying viscosity" begin
        # A closure reading a non-uniform backing field must produce a
        # different residual from a constant one — the smoke test that the
        # hook is wired up.
        U = (1f0, 0f0)
        N = (16, 16)
        νconst = fill(Float32(0.01), N .+ 2)
        νvary  = copy(νconst); νvary[8:end, :] .= Float32(0.05)

        a1 = WaterLily.Flow(N, U; T=Float32, ν=(I -> νconst[I]))
        a2 = WaterLily.Flow(N, U; T=Float32, ν=(I -> νvary[I]))
        for I in CartesianIndices(a1.u)
            v = sinpi((I.I[1] - 1) / N[1]) * cospi((I.I[2] - 1) / N[2])
            a1.u[I] = a2.u[I] = Float32(v)
        end
        Φ1, Φ2 = similar(a1.p), similar(a2.p)
        WaterLily.conv_diff!(a1.f, a1.u, Φ1, WaterLily.quick; ν=a1.ν)
        WaterLily.conv_diff!(a2.f, a2.u, Φ2, WaterLily.quick; ν=a2.ν)
        @test maximum(abs.(a1.f .- a2.f)) > 0
    end

    @testset "CFL with closure ν" begin
        U = (1f0, 0f0)
        N = (16, 16)
        νfield = fill(Float32(0.01), N .+ 2)
        νfield[2, 2] = Float32(0.5)   # interior spike — most restrictive
        a = WaterLily.Flow(N, U; T=Float32, ν=(I -> νfield[I]))
        a.u .= 1f0
        Δt = WaterLily.CFL(a)
        @test Δt < 1 / (5 * 0.5)
    end

    @testset "closure is stored by reference" begin
        # The closure is kept by reference, so mutating the array it wraps
        # is seen by the next conv_diff! without rebuilding the Flow.
        U = (1f0, 0f0)
        N = (16, 16)
        νfield = fill(Float32(0.01), N .+ 2)
        clos = I -> νfield[I]
        a = WaterLily.Flow(N, U; T=Float32, ν=clos)
        @test a.ν === clos
        for I in CartesianIndices(a.u)
            a.u[I] = Float32(sinpi((I.I[1] - 1) / N[1]))
        end
        Φ = similar(a.p)
        WaterLily.conv_diff!(a.f, a.u, Φ, WaterLily.quick; ν=a.ν)
        f0 = copy(a.f)
        νfield .*= 10f0                      # mutate the wrapped array in place
        WaterLily.conv_diff!(a.f, a.u, Φ, WaterLily.quick; ν=a.ν)
        @test maximum(abs.(a.f .- f0)) > 0
    end

end

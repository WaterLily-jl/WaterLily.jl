# Regression test for the scalar transport! helper.
#
# transport!(r, φ, u, Φ; D_diff, λ, perdir) computes the cell-centred
# conservative advection-diffusion residual for a passive scalar.

using Test
using WaterLily
using StaticArrays

@testset "transport!" begin

    @testset "uniform advection conserves total mass" begin
        # 2D: a smooth blob in a uniform stream. ∑ φ should not change
        # under transport! (the residual integrates to zero up to BC
        # fluxes, which for a blob in the interior is exactly zero).
        N = (32, 32)
        Ng = N .+ 2
        φ = zeros(Float32, Ng)
        # Gaussian blob centred at (16, 16) with std=3
        for I in CartesianIndices(φ)
            x, y = I.I .- 1.5
            φ[I] = exp(-((x-16)^2 + (y-16)^2) / 9)
        end
        # Uniform stream in +x: u_x = 1, u_y = 0, on the staggered grid
        u = zeros(Float32, (Ng..., 2))
        u[:, :, 1] .= 1f0   # x-velocity

        r = similar(φ); Φ = similar(φ)
        WaterLily.transport!(r, φ, u, Φ; D_diff=0f0)

        # Sum of residual over the interior should be the net flux through
        # all interior faces — which sums to zero for a closed interior.
        # (For a Gaussian centred well away from any boundary, this is
        # zero in exact arithmetic; allow Float32 rounding.)
        mass_change = sum(@view r[2:end-1, 2:end-1])
        @test abs(mass_change) < 1e-4
    end

    @testset "still fluid → no transport" begin
        N = (16, 16)
        Ng = N .+ 2
        φ = rand(Float32, Ng)
        u = zeros(Float32, (Ng..., 2))   # u = 0
        r = similar(φ); Φ = similar(φ)
        WaterLily.transport!(r, φ, u, Φ; D_diff=0f0)
        # No flux anywhere → r should be exactly zero
        @test all(iszero, r)
    end

    @testset "pure diffusion: r matches D·∇²φ on interior" begin
        # With u=0, transport! reduces to D · Laplacian(φ).
        N = (16, 16)
        Ng = N .+ 2
        φ = zeros(Float32, Ng)
        # Quadratic profile: φ = x²; ∇²φ = 2 → r = -∂_j(-D ∂_j φ) = D · 2
        # Wait — transport! computes r = -∂_j(uⱼφ - D ∂_j φ). With u=0:
        # r = -∂_j(-D ∂_j φ) = D · ∂_j ∂_j φ = D · ∇²φ. Sign: positive
        # diffusion lowers high points, so r at the peak should be negative.
        # For φ = x² with D > 0, ∇²φ = 2 (constant), so r = 2D everywhere.
        for I in CartesianIndices(φ)
            x = I.I[1] - 1.5
            φ[I] = Float32(x^2)
        end
        u = zeros(Float32, (Ng..., 2))
        D = 0.1f0
        r = similar(φ); Φ = similar(φ)
        WaterLily.transport!(r, φ, u, Φ; D_diff=D)
        # Check deeply interior cell: r should be 2D ≈ 0.2
        @test isapprox(r[8, 8], 2 * D; atol = 1e-4)
        @test isapprox(r[10, 10], 2 * D; atol = 1e-4)
    end

    @testset "moving step under upwind moves the right way" begin
        # 1D-ish: step function advecting in +x. After a single step
        # (φ_new = φ + dt * r), the step front should advance.
        N = (32, 16)
        Ng = N .+ 2
        φ = zeros(Float32, Ng)
        φ[1:16, :] .= 1f0   # left half
        u = zeros(Float32, (Ng..., 2))
        u[:, :, 1] .= 1f0
        r = similar(φ); Φ = similar(φ)
        WaterLily.transport!(r, φ, u, Φ; D_diff=0f0)
        dt = 0.1f0
        φ_new = φ .+ dt .* r
        # The mid-row should now have higher values on the right side of
        # the step than the original.
        @test sum(@view φ_new[17:20, 8]) > sum(@view φ[17:20, 8])
    end

    @testset "diffusivity field: array/closure match the scalar" begin
        # D_diff may be a Number, a cell-centred array, or a callable; a
        # constant array/closure must reproduce the scalar-D result, and a
        # varying field must change it.
        N = (16, 16); Ng = N .+ 2
        φ = zeros(Float32, Ng)
        for I in CartesianIndices(φ)
            φ[I] = Float32((I.I[1] - 1.5)^2)
        end
        u = zeros(Float32, (Ng..., 2))
        D = 0.1f0
        rs = similar(φ); Φs = similar(φ)
        ra = similar(φ); Φa = similar(φ)
        rc = similar(φ); Φc = similar(φ)
        Darr = fill(D, Ng)
        WaterLily.transport!(rs, φ, u, Φs; D_diff=D)
        WaterLily.transport!(ra, φ, u, Φa; D_diff=Darr)
        WaterLily.transport!(rc, φ, u, Φc; D_diff=(I -> @inbounds Darr[I]))
        @test maximum(abs.(rs .- ra)) ≤ 4 * eps(Float32) * maximum(abs.(rs))
        @test maximum(abs.(rs .- rc)) ≤ 4 * eps(Float32) * maximum(abs.(rs))
        Dvar = copy(Darr); Dvar[8:end, :] .= 0.5f0
        rv = similar(φ); Φv = similar(φ)
        WaterLily.transport!(rv, φ, u, Φv; D_diff=Dvar)
        @test maximum(abs.(rv .- rs)) > 0
    end

end

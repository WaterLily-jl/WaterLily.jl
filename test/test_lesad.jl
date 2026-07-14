# Implicit-LES dissipative-flux tests (les-ad branch). Self-contained (WaterLily + Test +
# StaticArrays + Random + ForwardDiff) so it runs without the optional GPUArrays/CUDA test deps.
# Auto-discovered by runtests.jl (test_*.jl); runnable standalone:  julia --project=. test/test_lesad.jl
# The udf advecting-velocity regression lives upstream in test_les.jl; the general udf
# machinery (body force, rotating frame) is covered in test_flow.jl.
using WaterLily, Test, StaticArrays, Random
using WaterLily: dissipative_flux!, _apply_dissipative_flux!, spatial_energy_rate, sgs!,
                 size_u, inside_u, cds, quick, ϕ, CI, dot, mom_project!
import WaterLily
import ForwardDiff

@testset "dissipative flux" begin
    Tt = Float64; Nn = 8
    # N=8 triple-periodic inviscid box + a discretely div-free field (λ=cds at construction —
    # post-#301 the convective scheme lives in flow.λ; sim_step! ignores a λ kwarg)
    sim = Simulation((Nn,Nn,Nn),(0,0,0),1.0; U=1.0, ν=0.0, perdir=(1,2,3), T=Tt, λ=cds)
    flow = sim.flow
    Random.seed!(1234)
    flow.u .= randn(Tt, size(flow.u)); BC!(flow.u, flow.uBC, false, flow.perdir)
    flow.Δt[end] = 1.0
    for _ in 1:2; mom_project!(flow, sim.pois, 1.0, 0.0); end
    u0 = copy(flow.u)
    nrm = sum(abs2, @inbounds(flow.u[Ii]) for Ii in inside_u(size_u(flow.u)[1]))

    # R0: cds energy-conserving base ≪ dissipative quick
    flow.u .= u0; r_cds   = spatial_energy_rate(flow; λ=cds,   ν=0.0)
    flow.u .= u0; r_quick = spatial_energy_rate(flow; λ=quick, ν=0.0)
    @test abs(r_cds)/nrm < 1e-4                     # cds adds ~no numerical dissipation
    @test r_quick < 0                               # quick is dissipative
    @test abs(r_cds) < 1e-2*abs(r_quick)

    # R1: P=1 dissipation ≤0, monotone, exactly linear, matches analytic -½Σ|U|β₁(Δu)²
    function analytic_p1_coeff(u, perdir)
        Nf,n = size_u(u); s = zero(eltype(u))
        for i in 1:n, j in 1:n, I in CartesianIndices(ntuple(k -> 2:Nf[k]-1, length(Nf)))
            U = ϕ(i, CI(I,j), u); du = WaterLily.∂(j, CI(I,i), u)   # du = Δ²
            s += abs(U)*du*du
        end
        -0.5*s
    end
    coeff = analytic_p1_coeff(u0, flow.perdir)
    diss_at(β1) = (flow.u .= u0; spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=Tt[β1]) - r_cds)
    βs = [0.05, 0.1, 0.2, 0.4]; diss = [diss_at(b) for b in βs]
    @test all(<(0), diss)                                          # dissipative
    @test issorted(diss; rev=true)                                 # monotone decreasing
    slopes = diss ./ βs
    @test maximum(slopes)-minimum(slopes) <= 1e-9*abs(coeff)       # exactly linear in β₁
    @test all(abs.(diss .- coeff.*βs) .<= 1e-7*max(1,abs(coeff)))  # matches analytic

    # R2: cross-check P=1 vs Smagorinsky sgs! (same sign; magnitude is field-dependent)
    Nf = size_u(flow.u)[1]; S = zeros(Tt, Nf..., 3, 3)
    smag(I; S, Cs, Δ) = @views (Cs*Δ)^2*sqrt(2*dot(S[I,:,:], S[I,:,:]))
    Cs = 0.17; Δ = 1.0   # Δ=h (cube-root-volume convention); β₁=2Cs² bridge assumes this
    flow.u .= u0; r_smag = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=sgs!, νₜ=smag, S=S, Cs=Cs, Δ=Δ) - r_cds
    flow.u .= u0; r_beta = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=Tt[2*Cs^2]) - r_cds
    @test r_smag < 0 && r_beta < 0
    @test 0.01 < r_beta/r_smag < 100

    # R3: ForwardDiff dL/dβ₁ (Dual scratch arrays) == finite-diff == analytic slope
    function L(βv)
        Tβ = eltype(βv); u = Tβ.(u0)
        f = zeros(Tβ, size(u)); σ = zeros(Tβ, size(u)[1:end-1])
        _apply_dissipative_flux!(f, σ, u, βv, (1,2,3))
        sum(@inbounds(u[Ii]*f[Ii]) for Ii in inside_u(size_u(u)[1]))
    end
    g_ad = ForwardDiff.derivative(b -> L([b]), 0.13)
    h = 1e-6; g_fd = (L([0.13+h]) - L([0.13-h]))/(2h)
    @test isfinite(g_ad) && abs(g_ad) > 0
    @test abs(g_ad-g_fd) <= 1e-5*max(1,abs(g_fd))
    @test abs(g_ad-coeff) <= 1e-8*max(1,abs(coeff))

    # R4: end-to-end sim_step! (β rides kwargs; cds base set at construction).
    # β=0 conserves KE up to the small temporal (RK) drift — this assertion catches a
    # dissipative base scheme sneaking in (quick decays KE by ≫1e-3 over these steps).
    tgv(i,x) = i==1 ? sin(2π*x[1]/Nn)*cos(2π*x[2]/Nn) : i==2 ? -cos(2π*x[1]/Nn)*sin(2π*x[2]/Nn) : 0.0
    KE(fl) = 0.5*sum(abs2, @inbounds(fl.u[Ii]) for Ii in inside_u(size_u(fl.u)[1]))
    function run_steps(β1; nsteps=15)
        s = Simulation((Nn,Nn,Nn),(0,0,0),1.0; U=1.0, ν=0.0, perdir=(1,2,3), T=Tt, λ=cds, u0=tgv)
        mom_project!(s.flow, s.pois, 1.0, 0.0); ke0 = KE(s.flow); ok = true
        for _ in 1:nsteps
            sim_step!(s; remeasure=false, udf=dissipative_flux!, β=Tt[β1])
            ok &= all(isfinite, s.flow.u)
        end
        ke0, KE(s.flow), ok
    end
    ke0_a, ke1_a, fin_a = run_steps(0.0); ke0_b, ke1_b, fin_b = run_steps(0.2)
    @test fin_a && fin_b                            # stays finite
    @test abs(ke1_a/ke0_a - 1) < 1e-3               # β=0 + cds ⇒ KE conserved (spatial EC)
    @test ke1_b < ke0_b                             # β>0 decays KE
    @test ke1_b < ke1_a                             # more dissipation than β=0

    # R5: telescoping ⇒ per-component momentum conservation (P=2)
    ff = zero(u0); σσ = zeros(Tt, size(u0)[1:end-1])
    _apply_dissipative_flux!(ff, σσ, u0, Tt[0.1, 0.03], (1,2,3))
    momresid = maximum(abs(sum(@inbounds(ff[I,c]) for I in CartesianIndices(ntuple(k->2:size_u(u0)[1][k]-1,3)))) for c in 1:3)
    @test momresid < 1e-10

    # R6: periodic wrap (P≥3) agrees with the direct P=1 path and stays finite
    flow.u .= u0; r_p1 = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=Tt[0.1])
    flow.u .= u0; r_p3 = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=Tt[0.1,0.0,0.0])
    flow.u .= u0; r_p4 = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=Tt[0.05,0.01,0.0,0.002])
    @test abs(r_p1-r_p3) <= 1e-9*max(1,abs(r_p1))   # P=3 wrap path == P=1 direct when β₂=β₃=0
    @test isfinite(r_p4)

    # R7: per-order dissipative sign — Δ⁴ injects for β₂>0, dissipates for β₂<0
    flow.u .= u0; r0   = spatial_energy_rate(flow; λ=cds, ν=0.0)
    flow.u .= u0; r_d4p = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=Tt[0.0, 0.1]) - r0
    flow.u .= u0; r_d4m = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=Tt[0.0,-0.1]) - r0
    @test r_d4p > 0
    @test r_d4m < 0

    # R8: ForwardDiff through the public dissipative_flux! on Dual fields (production AD pattern)
    function Lflow(βv)
        Tβ = eltype(βv)
        fl = (f = zeros(Tβ, size(u0)), σ = zeros(Tβ, size(u0)[1:end-1]), u = Tβ.(u0), perdir = (1,2,3))
        dissipative_flux!(fl, fl.u, zero(Tβ); β=βv, ε=Tβ(1e-3))
        sum(@inbounds(fl.u[Ii]*fl.f[Ii]) for Ii in inside_u(size_u(fl.u)[1]))
    end
    gfl_ad = ForwardDiff.derivative(b -> Lflow([b]), 0.13)
    gfl_fd = (Lflow([0.13+1e-6]) - Lflow([0.13-1e-6]))/(2e-6)
    @test isfinite(gfl_ad) && abs(gfl_ad) > 0
    @test abs(gfl_ad-gfl_fd) <= 1e-5*max(1,abs(gfl_fd))
end

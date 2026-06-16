# LES-extension tests for WaterLily. Self-contained (WaterLily + Test + StaticArrays +
# Random + ForwardDiff) so it runs without the optional GPUArrays/CUDA test deps.
# Included from runtests.jl and runnable standalone:  julia --project=. test/les_tests.jl
using WaterLily, Test, StaticArrays, Random
using WaterLily: dissipative_flux!, _apply_dissipative_flux!, spatial_energy_rate, sgs!,
                 size_u, inside_u, inside, loc, cds, quick, ϕ, CI, δ, dot, mom_project!
import WaterLily
import ForwardDiff

@testset "udf advecting velocity" begin
    # udf! supplies the velocity the convective flux uses each phase (u⁰ in the predictor,
    # projected u in the corrector). A 3-arg force!(flow,u,t) udf uses it; a 2-arg
    # force!(flow,t) udf is unchanged (applicable() fallback) — backward compatible.
    saw = Tuple{Float64,Float64}[]
    rec!(flow, u, t; kw...) = (push!(saw, (maximum(abs, @view u[inside_u(u),:]),
                                           maximum(abs, @view flow.u[inside_u(flow.u),:]))); nothing)
    sim = Simulation((16,16),(1.0,0.0),16; U=1.0, T=Float64, mem=Array)
    empty!(saw); sim_step!(sim; udf=rec!)
    @test saw[1][1] > 1e-8       # predictor udf sees nonzero u⁰ (the fix)
    @test saw[1][2] < 1e-8       # while flow.u interior is zeroed (the old bug source)
    @test saw[end][1] > 1e-8     # corrector udf sees the nonzero projected field

    # 2-arg force-only udf still runs unchanged
    NG = Ref(0)
    grav!(flow, t; g=0.5) = (WaterLily.@loop flow.f[Ii] += g over Ii in CartesianIndices(flow.f); NG[]+=1; nothing)
    sim2 = Simulation((16,16),(1.0,0.0),16; U=1.0, T=Float64, mem=Array)
    sim_step!(sim2; udf=grav!, g=0.5)
    @test NG[] == 2 && all(isfinite, sim2.flow.u)

    # the existing maintests udf testsets must still hold (replicated here on Array)
    function acceleratingFlow(N; use_g=false, T=Float64, perdir=(1,), jerk=4, mem=Array)
        UScale=√N; g(i,x,t)= i==1 ? t*jerk : 0.; !use_g && (g=nothing)
        Simulation((N,N),(UScale,0.),N; ν=0.001, g, Δt=0.001, perdir, T, mem), jerk
    end
    gravity!(flow, t; jerk=4) = for i ∈ 1:last(size(flow.f))
        WaterLily.@loop flow.f[I,i] += i==1 ? t*jerk : 0 over I ∈ CartesianIndices(Base.front(size(flow.f)))
    end
    N=8; simg,jerk = acceleratingFlow(N; use_g=true); sim_step!(simg,1.0)
    uF = simg.flow.uBC[1] + 0.5*jerk*WaterLily.time(simg)^2
    simu,_ = acceleratingFlow(N); sim_step!(simu,1.0; udf=gravity!, jerk=jerk)
    @test WaterLily.L₂(simu.flow.u[:,:,1].-uF)<1e-4 && WaterLily.L₂(simu.flow.u[:,:,2].-0)<1e-4
    @test WaterLily.L₂(simg.flow.u[:,:,1].-uF)<1e-4

    L=4; x₀=SA_F64[L,L]; ω=1/L
    vel(i,x,t)= begin s,c=sincos(ω*t); y=ω*(x-x₀); i==1 ? s*y[1]+c*y[2] : -c*y[1]+s*y[2] end
    cor(i,x,t)= i==1 ? 2ω*vel(2,x,t) : -2ω*vel(1,x,t); cen(i,x,t)=ω^2*(x-x₀)[i]; g(i,x,t)=cor(i,x,t)+cen(i,x,t)
    rotudf(a,t)=WaterLily.@loop a.f[Ii]+=g(last(Ii),loc(Ii,eltype(a.f)),t) over Ii in CartesianIndices(a.f)
    Nr=8; simgr=Simulation((Nr,Nr),vel,Nr; g, U=1, T=Float64); simr=Simulation((Nr,Nr),vel,Nr; U=1, T=Float64)
    sim_step!(simgr); sim_step!(simr; udf=rotudf)
    @test L₂(simgr.flow.p)==L₂(simr.flow.p)<3e-3
end

@testset "dissipative flux" begin
    Tt = Float64; Nn = 8
    # N=8 triple-periodic inviscid box + a discretely div-free field
    sim = Simulation((Nn,Nn,Nn),(0,0,0),1.0; U=1.0, ν=0.0, perdir=(1,2,3), T=Tt)
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
    Cs = 0.17; Δ = sqrt(3.0)
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

    # R4: end-to-end sim_step! (β rides kwargs); β=0 ~preserves KE, β>0 decays it
    tgv(i,x) = i==1 ? sin(2π*x[1]/Nn)*cos(2π*x[2]/Nn) : i==2 ? -cos(2π*x[1]/Nn)*sin(2π*x[2]/Nn) : 0.0
    KE(fl) = 0.5*sum(abs2, @inbounds(fl.u[Ii]) for Ii in inside_u(size_u(fl.u)[1]))
    function run_steps(β1; nsteps=15)
        s = Simulation((Nn,Nn,Nn),(0,0,0),1.0; U=1.0, ν=0.0, perdir=(1,2,3), T=Tt, uλ=tgv)
        mom_project!(s.flow, s.pois, 1.0, 0.0); ke0 = KE(s.flow); ok = true
        for _ in 1:nsteps
            sim_step!(s; remeasure=false, λ=cds, udf=dissipative_flux!, β=Tt[β1])
            ok &= all(isfinite, s.flow.u)
        end
        ke0, KE(s.flow), ok
    end
    _, ke1_a, fin_a = run_steps(0.0); ke0_b, ke1_b, fin_b = run_steps(0.2)
    @test fin_a && fin_b                            # stays finite
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

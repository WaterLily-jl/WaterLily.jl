# N=8 validation of the implicit-LES parametrized dissipative flux (dissipative_flux!).
# Run from the WaterLily package dir:   julia --project=. validate_dissipative_flux.jl
#
# Recipes (see memory/waterlily-flux-decision.md):
#   0  cds is the energy-conserving base: spatial_energy_rate(cds,β=0) ≈ 0  ≪ |quick|
#   1  dissipation monotone & ≤0 in β₁, matches analytic -½Σ|U_face|β₁(Δu)²
#   2  f^d(β₁=2Cs²) removes energy of the same sign/order as Smagorinsky sgs!(Cs)
#   3  ForwardDiff dL/dβ is finite and matches a central finite difference

using WaterLily
using WaterLily: dissipative_flux!, spatial_energy_rate, _apply_dissipative_flux!,
                 sgs!, size_u, inside_u, inside, cds, quick, ϕ, CI, δ, dot
import WaterLily
using Random, Printf
import ForwardDiff

const T = Float64
const N = 8
pass = Ref(true)
check(name, ok) = (pass[] &= ok; println(rpad(ok ? "  PASS " : "  FAIL ",8), name))

# ---- build an N=8 triple-periodic, inviscid sim and a discretely div-free field ----
sim  = Simulation((N,N,N), (0,0,0), 1.0; U=1.0, ν=0.0, perdir=(1,2,3), T=T)
flow = sim.flow
Random.seed!(1234)
flow.u .= randn(T, size(flow.u))
BC!(flow.u, flow.uBC, false, flow.perdir)
# one exact discrete projection -> div-free
flow.Δt[end] = 1.0
for _ in 1:2; WaterLily.mom_project!(flow, sim.pois, 1.0, 0.0); end
u0 = copy(flow.u)                                   # frozen div-free field
nrm = sum(abs2, @inbounds(flow.u[Ii]) for Ii in inside_u(size_u(flow.u)[1]))
maxdiv = maximum(abs(WaterLily.div(I, flow.u)) for I in inside(flow.p))
@printf("setup: N=%d, ‖u‖²=%.4g, max|div u|=%.3e\n", N, nrm, maxdiv)
println("="^64)

# ---------- RECIPE 0: cds energy-conserving base vs dissipative quick ----------
flow.u .= u0
r_cds   = spatial_energy_rate(flow; λ=cds,   ν=0.0)
flow.u .= u0
r_quick = spatial_energy_rate(flow; λ=quick, ν=0.0)
@printf("R0  cds  dE/dt = %+.3e   (rel %.2e)\n", r_cds, abs(r_cds)/nrm)
@printf("R0  quick dE/dt = %+.3e   (rel %.2e)\n", r_quick, abs(r_quick)/nrm)
check("R0a cds is EC (|dE/dt|/‖u‖² < 1e-4)",         abs(r_cds)/nrm < 1e-4)
check("R0b quick is dissipative (dE/dt < 0)",         r_quick < 0)
check("R0c cds ≪ quick (|r_cds| < 1e-2 |r_quick|)",   abs(r_cds) < 1e-2*abs(r_quick))
println("="^64)

# ---------- RECIPE 1: P=1 dissipation monotone, ≤0, matches analytic ----------
# analytic added rate = -½ Σ_faces |U_face| β₁ (Δⱼuᵢ)²  (β₁ factored out below)
function analytic_p1_coeff(u, perdir)
    Nf,n = size_u(u); s = zero(eltype(u))
    for i in 1:n, j in 1:n
        # ALL periodic j-faces 2:Ng-1 (the code now closes the seam face j=2);
        # telescoping dE=Σσ·Δ² holds with both endpoint cells in inside_u(N) ⇒ 2:Ng-1
        R = CartesianIndices(ntuple(k -> 2:Nf[k]-1, length(Nf)))
        for I in R
            U  = ϕ(i, CI(I,j), u)
            du = WaterLily.∂(j, CI(I,i), u)          # = uᵢ[I]-uᵢ[I-δⱼ] = Δ²
            s += abs(U)*du*du
        end
    end
    -0.5*s
end
coeff = analytic_p1_coeff(u0, flow.perdir)
println("R1  analytic slope d(dE/dt)/dβ₁ = ", @sprintf("%+.4e", coeff))
prev = 0.0; mono = true; matchA = true; slopes = Float64[]
for β1 in (0.0, 0.05, 0.1, 0.2, 0.4)
    flow.u .= u0
    r = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=T[β1])
    diss = r - r_cds                                 # isolate the f^d contribution
    pred = coeff*β1
    @printf("R1  β₁=%.2f  added dE/dt = %+.4e   analytic = %+.4e   Δ=%.2e\n",
            β1, diss, pred, abs(diss-pred))
    global mono   &= (diss <= prev + 1e-10)
    global matchA &= (abs(diss-pred) <= 1e-7*max(1,abs(pred)))
    β1 > 0 && push!(slopes, diss/β1)
    global prev = diss
end
lin = (maximum(slopes)-minimum(slopes)) <= 1e-9*abs(coeff)   # exactly linear in β₁
check("R1a dissipation monotone decreasing in β₁", mono)
check("R1b exactly linear in β₁ (constant slope)",  lin)
check("R1c matches analytic -½Σ|U|β₁(Δu)²",         matchA)
println("="^64)

# ---------- RECIPE 2: cross-check P=1 vs Smagorinsky sgs! ----------
Nf = size_u(flow.u)[1]
S  = zeros(T, Nf..., 3, 3)
smag(I; S, Cs, Δ) = @views (Cs*Δ)^2*sqrt(2*dot(S[I,:,:], S[I,:,:]))
Cs = 0.17; Δ = sqrt(3.0)
flow.u .= u0
r_smag = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=sgs!, νₜ=smag, S=S, Cs=Cs, Δ=Δ) - r_cds
flow.u .= u0
r_beta = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=T[2*Cs^2]) - r_cds
@printf("R2  Smagorinsky(Cs=%.2f) added dE/dt = %+.3e\n", Cs, r_smag)
@printf("R2  f^d(β₁=2Cs²=%.4f)    added dE/dt = %+.3e\n", 2*Cs^2, r_beta)
@printf("R2  ratio f^d/Smag = %.3f  (NB: |U_face| vs |S| scaling differs; magnitude\n", r_beta/r_smag)
println("                       match is field-dependent — definitive test is the HIT spectrum)")
check("R2a both dissipative (dE/dt < 0)", r_smag < 0 && r_beta < 0)
check("R2b ratio within 2 orders (sane scaling)", 0.01 < r_beta/r_smag < 100)
println("="^64)

# ---------- RECIPE 3: ForwardDiff dL/dβ₁ (Dual-typed flow) matches finite diff ----------
function L(βv)                                       # added KE rate as a function of β
    Tβ = eltype(βv)
    u = Tβ.(u0)
    f = zeros(Tβ, size(u)); σ = zeros(Tβ, size(u)[1:end-1])
    _apply_dissipative_flux!(f, σ, u, βv, (1,2,3))
    sum(@inbounds(u[Ii]*f[Ii]) for Ii in inside_u(size_u(u)[1]))
end
β1₀ = 0.13
g_ad = ForwardDiff.derivative(b -> L([b]), β1₀)
h = 1e-6; g_fd = (L([β1₀+h]) - L([β1₀-h]))/(2h)
@printf("R3  dL/dβ₁  AD = %+.6e   FD = %+.6e   (analytic %+.6e)\n", g_ad, g_fd, coeff)
check("R3a AD gradient finite & nonzero", isfinite(g_ad) && abs(g_ad) > 0)
check("R3b AD matches finite difference", abs(g_ad-g_fd) <= 1e-5*max(1,abs(g_fd)))
check("R3c AD matches analytic slope",    abs(g_ad-coeff) <= 1e-8*max(1,abs(coeff)))
println("="^64)

# ---------- RECIPE 4: end-to-end sim_step! integration (β rides kwargs) ----------
# smooth, stable, div-free 2D Taylor–Green IC so the (non-dissipative) cds base
# does not blow up at the grid scale; β must thread sim_step!→mom_step!→udf!.
tgv(i,x) = i==1 ?  sin(2π*x[1]/N)*cos(2π*x[2]/N) :
           i==2 ? -cos(2π*x[1]/N)*sin(2π*x[2]/N) : 0.0
KE(fl) = 0.5*sum(abs2, @inbounds(fl.u[Ii]) for Ii in inside_u(size_u(fl.u)[1]))
function run_steps(β1; nsteps=15)
    s = Simulation((N,N,N),(0,0,0),1.0; U=1.0, ν=0.0, perdir=(1,2,3), T=T, uλ=tgv)
    WaterLily.mom_project!(s.flow, s.pois, 1.0, 0.0)
    ke0 = KE(s.flow); ok = true
    for _ in 1:nsteps
        sim_step!(s; remeasure=false, λ=cds, udf=dissipative_flux!, β=T[β1])
        ok &= all(isfinite, s.flow.u)
    end
    return ke0, KE(s.flow), ok
end
ke0_a, ke1_a, fin_a = run_steps(0.0)      # β=0: ~energy preserving (cds, tiny temporal drift)
ke0_b, ke1_b, fin_b = run_steps(0.2)      # β>0: dissipative
@printf("R4  β=0.0: KE %.4f → %.4f   |   β=0.2: KE %.4f → %.4f\n", ke0_a,ke1_a, ke0_b,ke1_b)
check("R4a sim_step! stays finite (β threads through)", fin_a && fin_b)
check("R4b dissipative flux decays KE (β>0)",            ke1_b < ke0_b)
check("R4c β=0.2 removes more KE than β=0",              ke1_b < ke1_a)
println("="^64)

# ---------- RECIPE 5: telescoping ⇒ momentum conservation (per component) ----------
Nf2 = size_u(u0)[1]
ff = zero(u0); σσ = zeros(T, size(u0)[1:end-1])
WaterLily._apply_dissipative_flux!(ff, σσ, u0, T[0.1, 0.03], (1,2,3))   # P=2
momresid = maximum(abs(sum(@inbounds(ff[I,c]) for I in CartesianIndices(ntuple(k->2:Nf2[k]-1,3))))
                   for c in 1:3)
@printf("R5  max |Σ fᵈ| per component (P=2) = %.3e\n", momresid)
check("R5  momentum conserved to machine zero", momresid < 1e-10)
println("="^64)

# ---------- RECIPE 6: periodic wrap (P≥3) runs & agrees with direct path ----------
flow.u .= u0; r_p1 = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=T[0.1])
flow.u .= u0; r_p3 = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=T[0.1,0.0,0.0])
flow.u .= u0; r_p4 = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=T[0.05,0.01,0.0,0.002])
@printf("R6  P=1 β₁=0.1: %+.4e   P=3 β=[0.1,0,0]: %+.4e   (Δ=%.2e)\n", r_p1, r_p3, abs(r_p1-r_p3))
@printf("R6  P=4 β=[.05,.01,0,.002] dE/dt = %+.4e (finite=%s)\n", r_p4, isfinite(r_p4))
check("R6a P=3 wrap path == P=1 direct when β₂=β₃=0", abs(r_p1-r_p3) <= 1e-9*max(1,abs(r_p1)))
check("R6b P=4 (full Δ²…Δ⁸ wrap) finite",            isfinite(r_p4))
println("="^64)

# ---------- RECIPE 7: per-order dissipative sign (even p injects for β>0) ----------
flow.u .= u0; r0   = spatial_energy_rate(flow; λ=cds, ν=0.0)
flow.u .= u0; r_d4p = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=T[0.0, 0.1]) - r0
flow.u .= u0; r_d4m = spatial_energy_rate(flow; λ=cds, ν=0.0, udf=dissipative_flux!, β=T[0.0,-0.1]) - r0
@printf("R7  Δ⁴ added dE/dt:  β₂=+0.1 → %+.3e   β₂=-0.1 → %+.3e\n", r_d4p, r_d4m)
check("R7a Δ⁴ INJECTS for β₂>0 (sign=(-1)^(p+1) ⇒ +)", r_d4p > 0)
check("R7b Δ⁴ dissipates for β₂<0",                    r_d4m < 0)
println("="^64)

# ---------- RECIPE 8: ForwardDiff through the public dissipative_flux! on Dual fields ----------
# Production AD needs Dual-typed flow buffers (reconstruct Flow with T=eltype(β)); a
# lightweight Dual NamedTuple "flow" exercises that exact pattern through the udf entry.
function Lflow(βv)
    Tβ = eltype(βv)
    fl = (f = zeros(Tβ, size(u0)), σ = zeros(Tβ, size(u0)[1:end-1]), u = Tβ.(u0), perdir = (1,2,3))
    dissipative_flux!(fl, fl.u, zero(Tβ); β=βv, ε=Tβ(1e-3))   # (flow,u,t) signature
    sum(@inbounds(fl.u[Ii]*fl.f[Ii]) for Ii in inside_u(size_u(fl.u)[1]))
end
gfl_ad = ForwardDiff.derivative(b -> Lflow([b]), 0.13)
h = 1e-6; gfl_fd = (Lflow([0.13+h]) - Lflow([0.13-h]))/(2h)
@printf("R8  dL/dβ₁ through dissipative_flux!:  AD = %+.6e   FD = %+.6e\n", gfl_ad, gfl_fd)
check("R8a AD through public udf finite & nonzero", isfinite(gfl_ad) && abs(gfl_ad) > 0)
check("R8b AD == FD through public udf",            abs(gfl_ad-gfl_fd) <= 1e-5*max(1,abs(gfl_fd)))
println("="^64)

println(pass[] ? "ALL CHECKS PASSED ✓" : "SOME CHECKS FAILED ✗")
exit(pass[] ? 0 : 1)

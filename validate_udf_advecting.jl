# Regression test for the udf advecting-velocity fix (branch udf-advecting-velocity).
# The udf now receives the velocity field the convective flux uses in each phase
# (u⁰ in the predictor, the projected u in the corrector), so velocity-dependent
# udfs (SGS / dissipative flux) no longer operate on the zeroed predictor field.
# Backward-compatible: a force-only `(flow,t)` udf is unchanged (applicable() fallback).
#   julia --project=. validate_udf_advecting.jl
using WaterLily
using WaterLily: size_u, inside_u
pass = Ref(true); check(n,ok) = (pass[] &= ok; println(rpad(ok ? "  PASS " : "  FAIL ",8), n))

# velocity-dependent udf (3-arg): records max|interior| of the field it RECEIVED (u)
# and of flow.u, per call.
saw = Tuple{Float64,Float64}[]
function rec!(flow, u, t; kw...)
    push!(saw, (maximum(abs, @view u[inside_u(u), :]), maximum(abs, @view flow.u[inside_u(flow.u), :])))
    nothing
end
# force-only udf (2-arg): must still run via the applicable() fallback
const NG = Ref(0)
grav!(flow, t; g=0.5) = (WaterLily.@loop flow.f[Ii] += g over Ii in CartesianIndices(flow.f); NG[]+=1; nothing)

sim = Simulation((16,16),(1.0,0.0),16; U=1.0, T=Float64, mem=Array)
empty!(saw); sim_step!(sim; udf=rec!)
println("  udf calls: ", length(saw), "   (max|u_seen|, max|flow.u_interior|) = ", round.(Iterators.flatten(saw), digits=4))
check("predictor udf sees NONZERO field u⁰ (the fix)",            saw[1][1] > 1e-8)
check("predictor flow.u interior IS zeroed (the old bug source)", saw[1][2] < 1e-8)
check("corrector udf sees nonzero (projected) field",             saw[end][1] > 1e-8)

sim2 = Simulation((16,16),(1.0,0.0),16; U=1.0, T=Float64, mem=Array)
NG[]=0; sim_step!(sim2; udf=grav!, g=0.5)
check("force-only (flow,t) udf still runs (backward compat)", NG[] == 2 && all(isfinite, sim2.flow.u))

# --- backward-compat against the ACTUAL existing test udfs (from test/maintests.jl) ---
# (this clone's full suite needs GPUArrays/CUDA test deps it doesn't have, so we replicate
#  the two udf testsets here on CPU/Array to confirm the signature change didn't break them)
using StaticArrays
function acceleratingFlow(N; use_g=false, T=Float64, perdir=(1,), jerk=4, mem=Array)
    UScale = √N; g(i,x,t) = i==1 ? t*jerk : 0.; !use_g && (g = nothing)
    WaterLily.Simulation((N,N),(UScale,0.),N; ν=0.001, g, Δt=0.001, perdir, T, mem), jerk
end
gravity!(flow, t; jerk=4) = for i ∈ 1:last(size(flow.f))   # 2-arg force udf (unchanged signature)
    WaterLily.@loop flow.f[I,i] += i==1 ? t*jerk : 0 over I ∈ CartesianIndices(Base.front(size(flow.f)))
end
let N=8
    simg,jerk = acceleratingFlow(N; use_g=true); sim_step!(simg,1.0); ug = simg.flow.u
    uF = simg.flow.uBC[1] + 0.5*jerk*WaterLily.time(simg)^2
    simu,_ = acceleratingFlow(N); sim_step!(simu,1.0; udf=gravity!, jerk=jerk); uu = simu.flow.u
    check("existing 'increasing body force' testset: udf≡built-in g",
          WaterLily.L₂(uu[:,:,1].-uF)<1e-4 && WaterLily.L₂(uu[:,:,2].-0)<1e-4 &&
          WaterLily.L₂(ug[:,:,1].-uF)<1e-4)
end
let L=4, N=8
    x₀=SA_F64[L,L]; ω=1/L
    vel(i,x,t)= begin s,c=sincos(ω*t); y=ω*(x-x₀); i==1 ? s*y[1]+c*y[2] : -c*y[1]+s*y[2] end
    cor(i,x,t)= i==1 ? 2ω*vel(2,x,t) : -2ω*vel(1,x,t); cen(i,x,t)=ω^2*(x-x₀)[i]
    g(i,x,t)=cor(i,x,t)+cen(i,x,t)
    rotudf(a,t)=WaterLily.@loop a.f[Ii]+=g(last(Ii),loc(Ii,eltype(a.f)),t) over Ii in CartesianIndices(a.f) # 2-arg
    simg=Simulation((N,N),vel,N; g, U=1, T=Float64); sim=Simulation((N,N),vel,N; U=1, T=Float64)
    sim_step!(simg); sim_step!(sim; udf=rotudf)
    check("existing 'rotating reference frame' testset: udf≡g", L₂(simg.flow.p)==L₂(sim.flow.p)<3e-3)
end

println(pass[] ? "UDF-ADVECTING FIX OK ✓" : "FAILED ✗"); exit(pass[] ? 0 : 1)

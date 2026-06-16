# LES-extension tests for WaterLily. Self-contained (WaterLily + Test + StaticArrays)
# so it runs without the optional GPUArrays/CUDA test deps. Included from runtests.jl
# and runnable standalone:  julia --project=. test/les_tests.jl
using WaterLily, Test, StaticArrays
using WaterLily: size_u, inside_u, loc

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

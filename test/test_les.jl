using WaterLily: inside_u

# Regression test for the udf advecting-velocity fix (branch udf-advecting-velocity):
# `mom_step!` now passes the velocity field the convective flux is evaluated on to the udf —
# `a.u⁰` in the predictor (`a.u` is zeroed by `scale_u!`) and the projected `a.u` in the
# corrector — so SGS/eddy-viscosity udfs see a nonzero advecting field instead of the zeroed
# predictor velocity. The general udf machinery (body force, rotating frame) is already covered
# in test_flow.jl ("increasing body force", "rotating reference frame"); here we only check the
# new 3-arg advecting-velocity signature and that the 2-arg force-only fallback still dispatches.
@testset "udf advecting velocity" begin
    # 3-arg udf supplies the velocity the convective flux uses each phase
    saw = Tuple{Float64,Float64}[]
    rec!(flow, u, t; kw...) = (push!(saw, (maximum(abs, @view u[inside_u(u),:]),
                                           maximum(abs, @view flow.u[inside_u(flow.u),:]))); nothing)
    sim = Simulation((16,16),(1.0,0.0),16; U=1.0, T=Float64, mem=Array)
    empty!(saw); sim_step!(sim; udf=rec!)
    @test saw[1][1] > 1e-8       # predictor udf sees nonzero u⁰ (the fix)
    @test saw[1][2] < 1e-8       # while flow.u interior is zeroed (the old bug source)
    @test saw[end][1] > 1e-8     # corrector udf sees the nonzero projected field

    # 2-arg force-only udf still dispatches and runs once per phase (predictor + corrector)
    NG = Ref(0)
    grav!(flow, t; g=0.5) = (WaterLily.@loop flow.f[Ii] += g over Ii in CartesianIndices(flow.f); NG[]+=1; nothing)
    sim2 = Simulation((16,16),(1.0,0.0),16; U=1.0, T=Float64, mem=Array)
    sim_step!(sim2; udf=grav!, g=0.5)
    @test NG[] == 2 && all(isfinite, sim2.flow.u)
end

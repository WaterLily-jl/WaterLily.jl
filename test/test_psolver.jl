using WaterLily
using StaticArrays
using BiotSavartBCs
using LoggingExtras

"""Circle function"""
function circle(L=32;m=6,n=4,Re=80,U=1,T=Float32)
    radius, center = L/2, max(n*L/2,L)
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((m*L,n*L), (U,0), radius; ν=U*radius/Re, body, T)
end

include("../examples/TwoD_plots.jl")

# # # hydrostatic pressure force
# f1=[]; f2=[]; f3=[]; resolutions = [16,32,64,128,256,512]
# for N ∈ resolutions
#     a = Flow((N,N),(1.,0.);f=Array,T=Float32)
#     sdf(x,t) = √sum(abs2,x.-N÷2)-N÷4
#     map(x,t) = x.-SVector(t,0)
#     body = AutoBody(sdf,map)
#     WaterLily.measure!(a,body)
#     @inside a.p[I] = loc(0,I)[2]
#     # @inside a.p[I] = sdf(loc(0,I),0) >= 0 ? loc(0,I)[2] : 0
#     push!(f1,WaterLily.pressure_force(a,body)/(π*(N÷4)^2))
# end
# plot(title="Hydrostatic pressure force",xlabel="N",ylabel="force/πR²")
# plot!(resolutions,reduce(hcat,f1)[2,:],label="WaterLily.pressure_force(sim)")
# savefig("hydrostatic_force.png")

# make the sim
body = AutoBody((x,t)->√sum(abs2,x.-N÷2)-N÷4,(x,t)->x.-SVector(t,0))
sim = circle(64;m=24,n=16,Re=80,U=1,T=Float32)
# ml_ω = MLArray(sim.flow.σ)

# allows logging the pressure solver results
WaterLily.logger("test_psolver")

# intialize
t₀ = sim_time(sim); duration = 10; tstep = 0.1
forces_p = []; forces_ν = []

# step and plot
anim  = @animate for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U

        # update flow
        mom_step!(sim.flow,sim.pois)
        # biot_mom_step!(sim.flow,sim.pois,ml_ω)
        
        # pressure force
        force = -2WaterLily.pressure_force(sim)
        push!(forces_p,force)
        vforce = -2WaterLily.viscous_force(sim)
        push!(forces_ν,vforce)
        # update time
        t += sim.flow.Δt[end]
    end
  
    # print time step
    println("tU/L=",round(tᵢ,digits=4),",  Δt=",round(sim.flow.Δt[end],digits=3))
    a = sim.flow.σ;
    @inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(a[inside(a)],clims=(-10,10), legend=false); body_plot!(sim)
    contour!(sim.flow.p[inside(a)]',levels=range(-1,1,length=10),
             color=:black,linewidth=0.5,legend=false)
    # flood(sim.flow.p[inside(a)],clims=(-1,1)); body_plot!(sim)
    # plot!([100],[200],marker=:o,color=:red,markersize=2,legend=false)
end
gif(anim, "cylinder_Float32_FullV.gif", fps = 10)
time = cumsum(sim.flow.Δt[4:end-1])
forces_p = reduce(vcat,forces_p')
forces_ν = reduce(vcat,forces_ν')
plot(time/sim.L,forces_p[4:end,1]/(sim.L),label="pressure force")
plot!(time/sim.L,forces_ν[4:end,1]/(sim.L),label="viscous force")
xlabel!("tU/L"); ylabel!("force/L"); savefig("cylinder_force_Float32_FullV.png")
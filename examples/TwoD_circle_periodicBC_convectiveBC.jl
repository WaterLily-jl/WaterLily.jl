using WaterLily
using StaticArrays
using Plots
include("TwoD_plots.jl")
function circle(p=4;Re=250,mem=Array,U=1)
    # Define simulation size, geometry dimensions, viscosity
    L=2^p
    center,r = SA[3L,3L], L
    ν = U*L/Re

    # functions for the body
    norm2(x) = √sum(abs2,x)
    function sdf(x,t)
        norm2(SA[x[1]-center[1],mod(x[2]-6L,6L)-center[2]])-r
    end
    function map(x,t)
        x.-SA[0.,U*t/2]
    end
    # make a body
    body = AutoBody(sdf,map)

    # return sim
    Simulation((8L,6L),(U,0),L;ν,body,mem,perdir=(2,),exitBC=true)
end

sim = circle(5);

# intialize
t₀ = sim_time(sim)
duration = 40.0
tstep = 0.1

# step and write
@time @gif for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=true)

    # print time step
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ,clims=(-10,10),shift=(-0.5,-0.5)); body_plot!(sim)
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end

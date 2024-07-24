using WaterLily
using StaticArrays

include("TwoD_plots.jl")

function circle(n,m;κ=1.5,Re=250,U=1)
    # define a circle at the domain center
    radius = m/8
    body = AutoBody((x,t)->√sum(abs2, x .- (n/2,m/2)) - radius)

    # define time-varying body force `g` and periodic direction `perdir`
    accelScale, timeScale = U^2/2radius, κ*radius/U
    g(i,t) = i==1 ? -2accelScale*sin(t/timeScale) : 0
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, g, perdir=(1,))
end

function run_oscillating_flow(n=392, stop=20.)
    sim = circle(n,n)
    sim_step!(sim,0.1)

    @time @gif for tᵢ in range(0.,stop;step=0.2)
        println("tU/L=",round(tᵢ,digits=4))
        sim_step!(sim,tᵢ)
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I])<0.001,0.0,sim.flow.σ[I])
        flood(sim.flow.σ,shift=(-2,-1.5),clims=(-8,8), axis=([], false),
            cfill=:seismic,legend=false,border=:none,size=(n,n))
        body_plot!(sim)
    end
end

run_oscillating_flow()
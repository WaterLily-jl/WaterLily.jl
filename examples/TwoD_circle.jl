using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function circle(n,m;Re=250)
    # Set physical parameters
    U,R,center = 1., m/8., [m/2,m/2]
    ν=U*R/Re
    @show R,ν

    # Immerse a circle (change for other shapes)
    c = BDIM_coef(n+2,m+2,2) do xy
        norm2(xy .- center) - R  # signed distance function
    end

    # Initialize Simulation object
    u = zeros(n+2,m+2,2)
    a = Flow(u,c,[U,0.],ν=ν)
    b = MultiLevelPoisson(c)
    Simulation(U,R,a,b)
end

function sim_gif!(sim;duration=1,step=0.1,verbose=true)
    t₀ = round(sim_time(sim))
    t = range(t₀,t₀+duration;step)
    gr(show=false)
    @time @gif for tᵢ in t
        sim_step!(sim,tᵢ;verbose)
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ,shift=(-0.5,-0.5),clims=(-5,5))
    end
    return
end

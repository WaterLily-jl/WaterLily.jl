using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")
using SmoothLivePlot

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

function v_plot(i,v,t)
    sleep(0.001)
    plot(t[1:i],v[1:i],xlims=(first(t),last(t)),legend=false)
    scatter!(t[i:i],v[i:i])
    plot!(xaxis=("time"),yaxis=("v[I]"))
end

function sim_measure!(sim,I;Δt=1,step=0.1)
    t₀ = round(sim_time(sim))
    t = range(t₀,t₀+Δt;step)
    v = Vector{Float64}(undef,length(t))
    plt = @makeLivePlot v_plot(1,v,t)
    for i ∈ 1:length(t)
        sim_step!(sim,t[i])
        # simulation will always slightly overshoot t[i]
        # so linearly interpolate over last time step
        r = (WaterLily.sim_time(sim)-t[i])/sim.a.Δt[end]
        v[i] = sim.a.u[I]*(1-r)+sim.a.u⁰[I]*r
        modifyPlotObject!(plt,arg1=i,arg2=v)
    end
    t,v
end

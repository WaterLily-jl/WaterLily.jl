using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")
include("TwoD_circle.jl")
using SmoothLivePlot

function v_plot(t,v,i=length(t))
    sleep(0.001)
    plot(t[1:i],v[1:i],xlims=(first(t),last(t)),legend=false)
    scatter!(t[i:i],v[i:i])
    plot!(xaxis=("time"),yaxis=("cross-stream velocity"))
end
"""
    sim_measure!(sim, I; duration=1, step=0.1)

Example function to run a simulation `sim` over time `duration`,
measuring the velocity `u[I]` every `step` and visualize `using
SmoothLivePlot`. The function modifies `sim` and returns the time
and velocity history `t,v`.

# Examples
```jldoctest
julia> sim, I = circle(128,64,Re=250), CartesianIndex(60,40,2);
julia> t, v = sim_measure!(sim, I, duration=100);
```
"""
function sim_measure!(sim,I;duration=1,step=0.1)
    t₀ = round(sim_time(sim))
    t = range(t₀,t₀+duration;step)
    v = Vector{Float64}(undef,length(t))
    plt = @makeLivePlot v_plot(t,v,1)
    for i ∈ 1:length(t)
        sim_step!(sim,t[i])
        v[i] = sim.flow.u[I]
        modifyPlotObject!(plt,arg2=v,arg3=i)
    end
    t,v
end

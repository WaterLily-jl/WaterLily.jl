module WaterLily

include("util.jl")
export show,show_scaled,L₂,BC!

include("PoissonSys.jl")
export Poisson,PoissonSys,solve!,mult

include("GMG.jl")
export MultiLevelPS,solve!

include("Flow.jl")
export Flow,mom_step!

end # module

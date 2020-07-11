module WaterLily

include("util.jl")
export L₂,BC!,@inside,inside,δ

include("PoissonSys.jl")
export Poisson,PoissonSys,solve!,mult

include("GMG.jl")
export MultiLevelPS,solve!

include("Flow.jl")
export Flow,mom_step!

include("plot.jl")
export flood,body

end # module

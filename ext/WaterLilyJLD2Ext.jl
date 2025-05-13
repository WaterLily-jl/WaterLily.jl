module WaterLilyJLD2Ext

if isdefined(Base, :get_extension)
    using JLD2
else
    using ..JLD2
end

using WaterLily
import WaterLily: save!, load!

"""
    save!(fname, flow::Flow; dir="./")

Save the `flow::Flow` pressure, velocity, and time steps arrays into a JLD2-formatted binary file (HDF5 compatible).
"""
save!(fname, flow::Flow; dir="./") = jldsave(
    joinpath(dir, fname);
    p=Array(flow.p),
    u=Array(flow.u),
    Δt=flow.Δt
)
save!(fname, sim::AbstractSimulation; dir="./") = save!(fname, sim.flow; dir)

"""
    load!(flow::Flow, fname::String; dir="./")

Load pressure, velocity, and time steps arrays from a JLD2-formatted binary file `dir/fname` into `flow::Flow`.
"""
function load!(flow::Flow, fname; dir="./")
    obj = jldopen(joinpath(dir, fname))
    @assert size(flow.p) == size(obj["p"]) "Simulation size does not match the size of the JLD2-stored simulation."
    f = typeof(flow.p).name.wrapper
    flow.p .= obj["p"] |> f
    flow.u .= obj["u"] |> f
    empty!(flow.Δt)
    push!(flow.Δt, obj["Δt"]...)
    close(obj)
end
load!(sim::AbstractSimulation, fname; dir="./") = load!(sim.flow, fname; dir)

end # module

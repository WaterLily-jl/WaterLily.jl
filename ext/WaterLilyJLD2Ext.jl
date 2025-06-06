module WaterLilyJLD2Ext

using JLD2, WaterLily
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
save!(fname::String, sim::AbstractSimulation; dir="./") = save!(fname, sim.flow; dir)

"""
    save!(fname, meanflow::MeanFlow; dir="./")

Save the `meanflow::MeanFlow` time-averaged pressure, velocity, velocity-squared, and time arrays into a JLD2-formatted binary file (HDF5 compatible).
"""
save!(fname, meanflow::MeanFlow; dir="./") = jldsave(
    joinpath(dir, fname);
    P=Array(meanflow.P),
    U=Array(meanflow.U),
    UU=Array(meanflow.UU),
    t=meanflow.t
)

"""
    load!(flow::Flow; kwargs...)

Load pressure, velocity, and time steps arrays from a JLD2-formatted binary file.
Keyword arguments considered are `fname="WaterLily.jld2"` and `dir="./"`.
"""
function load!(flow::Flow; kwargs...)
    fname = get(Dict(kwargs), :fname, "WaterLily.jld2")
    dir = get(Dict(kwargs), :dir, "./")
    obj = jldopen(joinpath(dir, fname))
    @assert size(flow.p) == size(obj["p"]) "Simulation size does not match the size of the JLD2-stored simulation."
    f = typeof(flow.p).name.wrapper
    flow.p .= obj["p"] |> f
    flow.u .= obj["u"] |> f
    empty!(flow.Δt)
    push!(flow.Δt, obj["Δt"]...)
    close(obj)
end
load!(sim::AbstractSimulation, ::Val{:jld2}; kwargs...) = load!(sim.flow; kwargs...)


"""
    load!(meanflow::MeanFlow; kwargs...)

Load time-averaged pressure, velocity, velocity-square, and time arrays from a JLD2-formatted binary file.
Keyword arguments considered are `fname="WaterLilyMean.jld2"` and `dir="./"`.
"""
function load!(meanflow::MeanFlow; kwargs...)
    fname = get(Dict(kwargs), :fname, "WaterLilyMean.jld2")
    dir = get(Dict(kwargs), :dir, "./")
    obj = jldopen(joinpath(dir, fname))
    @assert size(meanflow.P) == size(obj["P"]) "Simulation size does not match the size of the JLD2-stored simulation."
    f = typeof(meanflow.P).name.wrapper
    meanflow.P .= obj["P"] |> f
    meanflow.U .= obj["U"] |> f
    empty!(meanflow.t)
    push!(meanflow.t, obj["t"]...)
    close(obj)
end

end # module

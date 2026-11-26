module WaterLilyJLD2Ext

using JLD2, WaterLily
import WaterLily: save!, load!
using WaterLily: SVector, loc, interp, CFL

"""
    save!(fname, flow::AbstractFlow; dir="./")

Save the `flow::AbstractFlow` pressure, velocity, and time steps arrays into a JLD2-formatted binary file (HDF5 compatible).
"""
save!(fname, flow::AbstractFlow; dir="./") = jldsave(
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
    UU=isnothing(meanflow.UU) ? nothing : Array(meanflow.UU),
    t=meanflow.t
)

"""
    load!(flow::AbstractFlow; fname="WaterLily.jld2", dir="./", scale=1)

Load pressure, velocity, and time steps arrays from a JLD2-formatted binary file, interpolated
onto `flow`'s grid when its resolution is `scale` times the stored one or its domain is larger.
"""
function load!(flow::AbstractFlow; fname="WaterLily.jld2", dir="./", scale=1, kwargs...)
    jldopen(joinpath(dir, fname)) do obj
        sz, p = size(flow.p), obj["p"]
        check_sizes(sz, size(p), scale)
        regrid!(flow.p, p, sz, scale)
        regrid!(flow.u, obj["u"], sz, scale)
        empty!(flow.Δt)
        push!(flow.Δt, (scale .* obj["Δt"][1:end-1])..., CFL(flow))
    end
    flow
end
load!(sim::AbstractSimulation, ::Val{:jld2}; kwargs...) = load!(sim.flow; kwargs...)

"""
    load!(meanflow::MeanFlow; fname="WaterLilyMean.jld2", dir="./", scale=1)

Load time-averaged pressure, velocity, velocity-square, and time arrays from a JLD2-formatted binary file,
interpolated as in `load!(flow::AbstractFlow)`.
"""
function load!(meanflow::MeanFlow; fname="WaterLilyMean.jld2", dir="./", scale=1, kwargs...)
    jldopen(joinpath(dir, fname)) do obj
        sz, P = size(meanflow.P), obj["P"]
        check_sizes(sz, size(P), scale)
        regrid!(meanflow.P, P, sz, scale)
        regrid!(meanflow.U, obj["U"], sz, scale)
        isnothing(meanflow.UU) || regrid!(meanflow.UU, obj["UU"], sz, scale)
        empty!(meanflow.t)
        push!(meanflow.t, (scale .* obj["t"])...)
    end
    meanflow
end
check_sizes(a, b, s) = s == 1 && a != b && @warn "Saved grid size $b differs from request size $a; interpolating."

function regrid!(a, b, sz, scale)
    size(a) == size(b) && scale == 1 && return copyto!(a, b)
    T, N = eltype(a), length(sz)
    b = copyto!(similar(a, size(b)), b)
    for c in CartesianIndices(size(b)[N+1:end])
        s, r = SVector{N,T}(count(==(d), Tuple(c))/2max(1, length(c)) for d in 1:N), inv(T(scale))
        x(I) = (loc(0, I, eltype(s)) - s) * r + s
        view(a, ntuple(_ -> :, N)..., c) .= interp.(x.(CartesianIndices(sz)), Ref(view(b, ntuple(_ -> :, N)..., c)))
    end
    a
end

end # module

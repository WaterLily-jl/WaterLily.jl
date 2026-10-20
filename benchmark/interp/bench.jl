label, csv = ARGS
gpu = label == "ka-gpu"
using WaterLily, StaticArrays, BenchmarkTools, Printf
import WaterLily: @loop, loc, _interp_clamp
gpu && (using CUDA; @assert CUDA.functional())
mem = gpu ? CUDA.CuArray : Array
sync() = gpu ? CUDA.synchronize() : nothing

function interp_master(x::SVector{D,T}, varr::AbstractArray{T}) where {D,T}
    @inline shift(i) = SVector{D,T}(ifelse(i==j,0.5,0.) for j in 1:D)
    SVector{D,T}(_interp_master(_interp_clamp(x + shift(i), size(varr)[1:D]), view(varr, ntuple(_->:, D)..., i)) for i in 1:D)
end
interp_master(x::SVector{D,T}, arr::AbstractArray{T,D}) where {D,T} = _interp_master(_interp_clamp(x, size(arr)), arr)
function _interp_master(x::SVector{D,T}, arr::AbstractArray{T,D}) where {D,T}
    x = x .+ 1.5f0; i = floor.(Int,x); y = x.-i
    I = CartesianIndex(i...); R = I:I+oneunit(I)
    s = zero(T)
    @fastmath @inbounds @simd for J in R
        weight = prod(@. ifelse(J.I==I.I,1-y,y))
        s += arr[J]*weight
    end
    return s
end
@inline q(I, T) = loc(0, I, T) .- T(0.37)
@inline total(v::SVector) = sum(v); @inline total(v::Number) = v
kern!(out, arr, f) = @loop out[I] = total(f(q(I, eltype(arr)), arr)) over I ∈ CartesianIndices(out)

device = gpu ? CUDA.name(CUDA.device()) : Sys.cpu_info()[1].model
open(csv, "w") do io
    println(io, "# $label: $device, Julia $VERSION, backend=$(WaterLily.backend), nthreads=$(Threads.nthreads())")
    for (case, N) in (("2D 2048²", (2050, 2050)), ("3D 192×128²", (194, 130, 130)))
        D = length(N); out = mem(zeros(Float32, N...))
        for (kind, arr) in (("scalar", mem(rand(Float32, N...))), ("vector", mem(rand(Float32, N..., D))))
            t0, ref = 0.0, nothing
            for (variant, f) in (("master", interp_master), ("branch", WaterLily.interp))
                t = @belapsed (kern!($out, $arr, $f); sync()) seconds=1
                r = Array(out); ref === nothing && ((ref, t0) = (r, t))
                println(io, join((case, kind, variant, 1e3t, t0 / t, maximum(abs, r .- ref)), ','))
            end
        end
    end
end

module WaterLilyCUDAExt

using CUDA, WaterLily

"""
    __init__()

Asserts CUDA is functional when loading this extension.
"""
__init__() = @assert CUDA.functional()

# `@fastmath` sends `Dual` arithmetic through Base fallbacks that Julia stops inlining beyond a few partials (a call per
# operation in the kernel), so kernels over `Dual`s are compiled with everything inlined
WaterLily.kernel_backend(b::CUDABackend,::Type{<:WaterLily.ForwardDiff.Dual}) =
    CUDABackend(;(f=>getfield(b,f) for f in fieldnames(CUDABackend))...,always_inline=true)

end # module

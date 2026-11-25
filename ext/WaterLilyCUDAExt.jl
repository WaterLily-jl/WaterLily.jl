module WaterLilyCUDAExt

using CUDA, WaterLily

"""
    __init__()

Asserts CUDA is functional when loading this extension.
"""
__init__() = @assert CUDA.functional()

# Julia stops inlining `@fastmath` `Dual` arithmetic beyond 5 partials, so compile these kernels fully inlined
WaterLily.kernel_backend(b::CUDABackend,::Type{<:WaterLily.ForwardDiff.Dual}) = WaterLily.setproperties(b,(always_inline=true,))

end # module

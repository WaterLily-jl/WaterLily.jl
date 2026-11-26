module WaterLilyCUDAExt

using CUDA, WaterLily

"""
    __init__()

Asserts CUDA is functional when loading this extension.
"""
__init__() = @assert CUDA.functional()

WaterLily.kernel_backend(b::CUDABackend,::Type{<:WaterLily.ForwardDiff.Dual}) = WaterLily.setproperties(b,(always_inline=true,))

end # module

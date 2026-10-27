module WaterLilyCUDAExt

using CUDA, WaterLily
using LinearAlgebra: ⋅

"""
    __init__()

Asserts CUDA is functional when loading this extension.
"""
__init__() = @assert CUDA.functional()

# CUDA.jl has a dot kernel for views of CuArrays, faster than the generic GPU mapreduce in perdot
WaterLily.perdot(::CUDABackend,a,b,R) = @view(a[R])⋅@view(b[R])

# ImplicitGlobalGrid packs and unpacks each side of a halo on its own stream, and since 6.3
# CUDA.jl makes each launch on an array wait (on the host) for the stream it last ran on.
# The MPI extension already synchronizes the task stream before and IGG's streams after.
@static if isdefined(CUDA, :enable_synchronization!)
    WaterLily._no_stream_sync!(a::CuArray) = CUDA.enable_synchronization!(a, false)
end

end # module

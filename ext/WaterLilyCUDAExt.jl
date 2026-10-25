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

end # module

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

# MPI halos stage through page-locked host memory, which the pack and unpack kernels reach
# through its device mapping: no copies, and MPI is given host memory (with ImplicitGlobalGrid,
# staging through the host was faster than CUDA-aware ParaStation MPI on JUPITER's GH200s)
WaterLily._halo_buffers(::CuArray, T, n) = (b = CuVector{T,CUDA.HostMemory}(undef, n); (b, unsafe_wrap(Array, b)))

end # module

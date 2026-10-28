module WaterLilyCUDAExt

using CUDA, WaterLily
using LinearAlgebra: ⋅

"""
    __init__()

Asserts CUDA is functional when loading this extension, and registers how an MPI rank
picks one of the node's GPUs (`init_waterlily_mpi`).
"""
function __init__()
    @assert CUDA.functional()
    push!(WaterLily._device_selectors, i -> CUDA.device!(i % length(CUDA.devices())))  # MPI ranks of a node
end

# CUDA.jl has a dot kernel for views of CuArrays, faster than the generic GPU mapreduce in perdot
WaterLily.perdot(::CUDABackend,a,b,R) = @view(a[R])⋅@view(b[R])

# MPI halos staged through page-locked host memory, which the pack and unpack kernels reach
# through its device mapping: no copies, and MPI is given host memory
WaterLily._host_buffers(::CuArray, T, n) = (b = CuVector{T,CUDA.HostMemory}(undef, n); (b, unsafe_wrap(Array, b)))

end # module

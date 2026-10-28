module WaterLilyAMDGPUExt

using AMDGPU, WaterLily

"""
    __init__()

Asserts AMDGPU is functional when loading this extension.
"""
__init__() = @assert AMDGPU.functional()

# MPI halos staged through host memory, registered (page-locked) and mapped into the device for
# the pack and unpack kernels: no copies, and MPI is given host memory
WaterLily._host_buffers(::ROCArray, T, n) = (h = Vector{T}(undef, n); (unsafe_wrap(ROCArray, pointer(h), (n,)), h))

end # module

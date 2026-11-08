module WaterLilyAMDGPUExt

using AMDGPU, WaterLily

"""
    __init__()

Asserts AMDGPU is functional when loading this extension, and registers how an MPI rank
picks one of the node's GPUs (`init_waterlily_mpi`).
"""
function __init__()
    @assert AMDGPU.functional()
    push!(WaterLily._device_selectors, (l, n) -> AMDGPU.device_id!(WaterLily._node_gpu(l, n, length(AMDGPU.devices())) + 1))
end

# MPI halos staged through host memory, registered (page-locked) and mapped into the device for
# the pack and unpack kernels: no copies, and MPI is given host memory
WaterLily._host_buffers(::ROCArray, T, n) = (h = Vector{T}(undef, n); (unsafe_wrap(ROCArray, pointer(h), (n,)), h))

end # module

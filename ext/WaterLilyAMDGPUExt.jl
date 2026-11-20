module WaterLilyAMDGPUExt

using AMDGPU, WaterLily

"""
    __init__()

Asserts AMDGPU is functional when loading this extension, and registers the GPU selection of MPI ranks.
"""
function __init__()
    @assert AMDGPU.functional()
    push!(WaterLily._device_selectors, (l, n) -> AMDGPU.device_id!(WaterLily._node_gpu(l, n, length(AMDGPU.devices())) + 1))
end

# MPI halos staged through host memory registered and mapped into the device
WaterLily._host_buffers(::ROCArray, T, n) = (h = Vector{T}(undef, n); (unsafe_wrap(ROCArray, pointer(h), (n,)), h))

end # module

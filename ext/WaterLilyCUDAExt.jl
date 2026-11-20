module WaterLilyCUDAExt

using CUDA, WaterLily

"""
    __init__()

Asserts CUDA is functional when loading this extension, and registers the GPU selection of MPI ranks.
"""
function __init__()
    @assert CUDA.functional()
    push!(WaterLily._device_selectors, (l, n) -> CUDA.device!(WaterLily._node_gpu(l, n, length(CUDA.devices()))))
end

# MPI halos staged through page-locked host memory, mapped into the device
WaterLily._host_buffers(::CuArray, T, n) = (b = CuVector{T,CUDA.HostMemory}(undef, n); (b, unsafe_wrap(Array, b)))

# no implicit wait for an array's last stream when IGG's streams pack and unpack it (WaterLilyIGGExt syncs itself)
@static if isdefined(CUDA, :enable_synchronization!)
    WaterLily._no_stream_sync!(a::CuArray) = CUDA.enable_synchronization!(a, false)
end

end # module

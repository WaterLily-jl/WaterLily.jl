module WaterLilyCUDAExt

using CUDA, WaterLily

"""
    __init__()

Asserts CUDA is functional when loading this extension, and registers how an MPI rank
picks one of the node's GPUs (`init_waterlily_mpi`).
"""
function __init__()
    @assert CUDA.functional()
    push!(WaterLily._device_selectors, (l, n) -> CUDA.device!(WaterLily._node_gpu(l, n, length(CUDA.devices()))))
end

# MPI halos staged through page-locked host memory, pack/unpack use it with device mapping (no copies, MPI given host memory)
WaterLily._host_buffers(::CuArray, T, n) = (b = CuVector{T,CUDA.HostMemory}(undef, n); (b, unsafe_wrap(Array, b)))

# IGG packs/unpacks halos on its own stream, and (since 6.3) CUDA.jl makes each launch on an array wait (on the host)
# for the stream it last ran on. WaterLilyIGGExt already synchronizes the task stream before, IGG streams after.
@static if isdefined(CUDA, :enable_synchronization!)
    WaterLily._no_stream_sync!(a::CuArray) = CUDA.enable_synchronization!(a, false)
end

end # module

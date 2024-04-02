module WaterLilyCUDAExt

if isdefined(Base, :get_extension)
    using CUDA
else
    using ..CUDA
end

using WaterLily
import WaterLily: L₂

"""
    __init__()

Asserts CUDA is functional when loading this extension.
"""
__init__() = @assert CUDA.functional()

CUDA.allowscalar(false) # disallow scalar operations on GPU

"""
    L₂(a)

L₂ norm of CUDA array `a` excluding ghosts.
"""
L₂(a::CuArray,R::CartesianIndices=inside(a)) = mapreduce(abs2,+,@inbounds(a[R]))

end # module

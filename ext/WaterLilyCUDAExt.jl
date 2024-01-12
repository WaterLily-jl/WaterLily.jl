module WaterLilyCUDAExt

if isdefined(Base, :get_extension)
    using CUDA
else
    using ..CUDA
end

using WaterLily
import WaterLily: L₂

__init__() = @assert CUDA.functional()

CUDA.allowscalar(false)

L₂(a::CuArray,R::CartesianIndices=inside(a)) = mapreduce(abs2,+,@inbounds(a[R]))

end # module

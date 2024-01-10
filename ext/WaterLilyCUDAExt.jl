module WaterLilyCUDAExt

if isdefined(Base, :get_extension)
    using CUDA: CuArray, functional, allowscalar
else
    using ..CUDA: CuArray, functional, allowscalar
end

using WaterLily
import WaterLily: L₂

@assert functional()
allowscalar(false)
arrays_test() = [Array, CuArray]
L₂(a::CuArray,R::CartesianIndices=inside(a)) = mapreduce(abs2,+,@inbounds(a[R]))

end

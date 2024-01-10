module WaterLilyAMDGPUExt

if isdefined(Base, :get_extension)
    using AMDGPU: ROCArray, functional, allowscalar
else
    using ..AMDGPU: ROCArray, functional, allowscalar
end

using WaterLily
import WaterLily: L₂

@assert AMDGPU.functional()
AMDGPU.allowscalar(false)
arrays_test() = [Array, ROCArray]
L₂(a::ROCArray,R::CartesianIndices=inside(a)) = mapreduce(abs2,+,@inbounds(a[R]))

end

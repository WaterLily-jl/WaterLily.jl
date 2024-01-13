module WaterLilyAMDGPUExt

if isdefined(Base, :get_extension)
    using AMDGPU
else
    using ..AMDGPU
end

using WaterLily
import WaterLily: L₂

__init__() = @assert AMDGPU.functional()

AMDGPU.allowscalar(false)

L₂(a::ROCArray,R::CartesianIndices=inside(a)) = mapreduce(abs2,+,@inbounds(a[R]))

end # module

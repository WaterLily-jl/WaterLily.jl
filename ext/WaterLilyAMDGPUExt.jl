module WaterLilyAMDGPUExt

if isdefined(Base, :get_extension)
    using AMDGPU
else
    using ..AMDGPU
end

using WaterLily
import WaterLily: L₂

"""
    __init__()

Asserts AMDGPU is functional when loading this extension.
"""
__init__() = @assert AMDGPU.functional()

"""
    L₂(a)

L₂ norm of ROCArray `a` excluding ghosts.
"""
L₂(a::ROCArray,R::CartesianIndices=inside(a)) = mapreduce(abs2,+,@inbounds(a[R]))

end # module

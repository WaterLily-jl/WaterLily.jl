module WaterLilyOpenCLExt

if isdefined(Base, :get_extension)
    using OpenCL
else
    using ..OpenCL
end

using WaterLily
import WaterLily: L₂

"""
    L₂(a)

L₂ norm of OpenCL array `a` excluding ghosts.
"""
L₂(a::CLArray,R::CartesianIndices=inside(a)) = mapreduce(abs2,+,@inbounds(a[R]))

end # module

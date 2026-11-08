module WaterLilyAMDGPUExt

using AMDGPU, WaterLily

"""
    __init__()

Asserts AMDGPU is functional when loading this extension.
"""
__init__() = @assert AMDGPU.functional()

end # module

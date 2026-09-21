module WaterLilyCUDAExt

using CUDA, WaterLily

"""
    __init__()

Asserts CUDA is functional when loading this extension.
"""
__init__() = @assert CUDA.functional()

end # module

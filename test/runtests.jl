using WaterLily
using Test
using StaticArrays

check_compiler(compiler,parse_str) = try occursin(parse_str, read(`$compiler --version`, String)) catch _ false end
_cuda = true #check_compiler("nvcc","release")
_rocm = check_compiler("hipcc","version")
_cuda && using CUDA
_rocm && using AMDGPU
function setup_backends()
    arrays = [Array]
    _cuda && CUDA.functional() && push!(arrays, CUDA.CuArray)
    _rocm && AMDGPU.functional() && push!(arrays, AMDGPU.ROCArray)
    return arrays
end

arrays = setup_backends()
Threads.nthreads() > 1 ? include("maintests.jl") : include("alloctest.jl")

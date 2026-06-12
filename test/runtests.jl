using WaterLily
using Test
using StaticArrays

check_compiler(compiler,parse_str) = try occursin(parse_str, read(`$compiler --version`, String)) catch _ false end
_cuda = check_compiler("nvcc","release")
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

# MPI tests run as a separate testset that spawns child processes via mpiexec.
# Skipped on Windows (no system MPI in CI matrix) and via WATERLILY_SKIP_MPI=1.
if Sys.isunix() && get(ENV, "WATERLILY_SKIP_MPI", "0") != "1"
    include("mpitests.jl")
end

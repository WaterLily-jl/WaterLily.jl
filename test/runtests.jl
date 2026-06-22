using WaterLily, Test, StaticArrays, GPUArrays

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

#=
Test suite chosen by WaterLily `backend` preference (LocalPreferences.toml):
    Main sets (KernelAbstractions):
        core util poisson flow bodies forwarddiff metrics simulation ioext
    Allocations tests (SIMD): alloc
Within a suite, select set(s) with the WATERLILY_TEST environment variable (defaults to "all")
Run single sets locally with e.g.
   WATERLILY_TEST=poisson,bodies julia --project -e 'using Pkg; Pkg.test()'
=#
const WATERLILY_TEST = get(ENV, "WATERLILY_TEST", "all")

function run_set(file)
    tests_name = split(file, '_')[2][1:end-3]
    (WATERLILY_TEST == "all" || occursin(tests_name, WATERLILY_TEST)) || return
    include(file)
end

WaterLily.check_nthreads()
if backend == "KernelAbstractions"
    @info "Main tests with backends: $(join(arrays,", "))"
    include("helper.jl")
    run_set("test_core.jl")
    run_set("test_util.jl")
    run_set("test_poisson.jl")
    run_set("test_flow.jl")
    run_set("test_bodies.jl")
    run_set("test_forwarddiff.jl")
    run_set("test_metrics.jl")
    run_set("test_simulation.jl")
    run_set("test_ioext.jl")
else # backend == "SIMD"
    @info "Allocation tests"
    run_set("test_alloc.jl")
end
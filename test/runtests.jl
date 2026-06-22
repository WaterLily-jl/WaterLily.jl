using WaterLily, Test, StaticArrays, GPUArrays
import Pkg

# WATERLILY_BACKENDS selects array backends for tests: cpu|cuda|amdgpu|all
# Defaults to "cpu" (GPU backends are opt-in). Combinations like "cpu,cuda" are allowed
const WATERLILY_BACKENDS = filter(!isempty, strip.(split(lowercase(get(ENV, "WATERLILY_BACKENDS", "cpu")), ',')))
(isempty(WATERLILY_BACKENDS) || any(b -> b ∉ ("cpu","cuda","amdgpu","all"), WATERLILY_BACKENDS)) &&
    throw(ArgumentError("WATERLILY_BACKENDS must be a comma-separated list of cpu|cuda|amdgpu|all, got \"$(get(ENV, "WATERLILY_BACKENDS", ""))\""))

# A backend is requested via WATERLILY_BACKENDS, then confirmed at run time by
# CUDA/AMDGPU.functional() (a usable device + driver)
# A backend that cannot be installed / or not functional is skipped
_cpu = any(b -> b in ("cpu","all"), WATERLILY_BACKENDS)
_cuda = any(b -> b in ("cuda","all"), WATERLILY_BACKENDS)
_amdgpu = any(b -> b in ("amdgpu","all"), WATERLILY_BACKENDS)
if _cuda
    try
        Pkg.add("CUDA")
        using CUDA
        global _cuda = CUDA.functional()
    catch e
        @warn "WATERLILY_BACKENDS requested cuda but CUDA could not be installed/loaded; skipping" exception=e
        global _cuda = false
    end
end
if _amdgpu
    try
        Pkg.add("AMDGPU")
        using AMDGPU
        global _amdgpu = AMDGPU.functional()
    catch e
        @warn "WATERLILY_BACKENDS requested amdgpu but AMDGPU could not be installed/loaded; skipping" exception=e
        global _amdgpu = false
    end
end
function setup_backends()
    arrays = []
    _cpu && push!(arrays, Array)
    _cuda && push!(arrays, CUDA.CuArray)
    _amdgpu && push!(arrays, AMDGPU.ROCArray)
    isempty(arrays) && throw(ArgumentError("No functional backend available"))
    return arrays
end
arrays = setup_backends()

#=
Test suite chosen by WaterLily `backend` preference (LocalPreferences.toml):
    Main sets (KernelAbstractions):
        core util poisson flow bodies forwarddiff metrics simulation ioext
    Allocations tests (SIMD): alloc
Within a suite, select set(s) with the WATERLILY_TEST environment variable (defaults to "all"),
and limit the array backends with WATERLILY_BACKENDS (comma-separated cpu|cuda|amdgpu|all; see top)
Run single sets locally with e.g.
   WATERLILY_TEST=poisson,bodies WATERLILY_BACKENDS=cpu julia --project -e 'using Pkg; Pkg.test()'
=#
const WATERLILY_TEST = get(ENV, "WATERLILY_TEST", "all")

function run_set(file)
    tests_name = split(file, '_')[2][1:end-3]
    (WATERLILY_TEST == "all" || occursin(tests_name, WATERLILY_TEST)) || return
    include(file)
end

WaterLily.check_nthreads()
if backend == "KernelAbstractions"
    @testset verbose=true "WaterLily.jl" begin
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
    end
else # backend == "SIMD"
    @testset verbose=true "WaterLily.jl allocations" begin
        run_set("test_alloc.jl")
    end
end
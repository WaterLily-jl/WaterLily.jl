using WaterLily
import Pkg, ParallelTestRunner

#=
Parallel test runner (ParallelTestRunner.jl): every test_*.jl in this directory runs in
its own isolated module, in parallel across (single-threaded) worker processes.

Which suite runs is set by the WaterLily `backend` preference (LocalPreferences.toml):
    "KernelAbstractions" -> main sets (every test_*.jl except test_alloc.jl)
    "SIMD"               -> allocation tests (test_alloc.jl) only

Array backends: WATERLILY_BACKENDS=cpu|cuda|amdgpu|all (comma-separated, default "cpu").
GPU backends are opt-in and installed on demand (they are not test dependencies).

Run a subset by passing set name(s) (matched with startswith) and/or runner flags
(--jobs=N, --list, --verbose, --quickfail), e.g.
    julia --project -e 'using Pkg; Pkg.test(test_args=["poisson","flow"])'
=#

# --- array backends: request via WATERLILY_BACKENDS, install requested GPUs on demand ---
const WATERLILY_BACKENDS = filter(!isempty, strip.(split(lowercase(get(ENV, "WATERLILY_BACKENDS", "cpu")), ',')))
(isempty(WATERLILY_BACKENDS) || any(b -> b ∉ ("cpu","cuda","amdgpu","all"), WATERLILY_BACKENDS)) &&
    throw(ArgumentError("WATERLILY_BACKENDS must be a comma-separated list of cpu|cuda|amdgpu|all, got \"$(get(ENV, "WATERLILY_BACKENDS", ""))\""))
_cpu    = any(b -> b in ("cpu","all"),    WATERLILY_BACKENDS)
_cuda   = any(b -> b in ("cuda","all"),   WATERLILY_BACKENDS)
_amdgpu = any(b -> b in ("amdgpu","all"), WATERLILY_BACKENDS)

# GPU packages are not test deps: install the requested ones once here in the main process
# (workers share this project) so the sandboxes can `using` them; functional() is checked
# per sandbox. A backend that cannot be installed is skipped.
if _cuda
    try Pkg.add("CUDA") catch e; @warn "requested cuda but CUDA could not be installed; skipping" exception=e; global _cuda = false end
end
if _amdgpu
    try Pkg.add("AMDGPU") catch e; @warn "requested amdgpu but AMDGPU could not be installed; skipping" exception=e; global _amdgpu = false end
end

# --- discover test sets: every test_*.jl, gated by the WaterLily `backend` preference ---
const TESTDIR = @__DIR__
is_test_file(f) = startswith(f, "test_") && endswith(f, ".jl")
setname(f) = f[6:end-3]                                    # "test_core.jl" -> "core"
wanted(name) = backend == "SIMD" ? name == "alloc" : name != "alloc"
testsuite = Dict{String,Expr}()
for (root, _dirs, files) in walkdir(TESTDIR), f in filter(is_test_file, files)
    name = setname(f)
    wanted(name) && (testsuite[name] = :(include($(joinpath(root, f)))))
end

# --- per-sandbox setup (runs in each isolated module): imports, the `arrays` list, helpers ---
const init_code = quote
    using WaterLily, Test, StaticArrays, GPUArrays
    arrays = []
    $(_cpu)    && push!(arrays, Array)
    $(_cuda)   && using CUDA
    $(_amdgpu) && using AMDGPU
    $(_cuda)   && CUDA.functional()   && push!(arrays, CUDA.CuArray)
    $(_amdgpu) && AMDGPU.functional() && push!(arrays, AMDGPU.ROCArray)
    isempty(arrays) && error("No functional backend available")
    include($(joinpath(TESTDIR, "helper.jl")))
end

ParallelTestRunner.runtests(WaterLily, ARGS; testsuite, init_code)

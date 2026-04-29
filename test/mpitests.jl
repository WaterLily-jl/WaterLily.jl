# MPI test orchestrator. Spawns child Julia processes via `MPI.mpiexec()` to
# run `mpitests_workers.jl` in three roles:
#
#   - "parity_serial"        : plain Julia, serial reference
#   - "parity_parallel_cyl"  : np=NP_2D, 2D moving cylinder
#   - "parity_parallel_sph"  : np=NP_3D, 3D sphere
#   - "unit"                 : np=NP_2D, rank-aware unit tests
#
# Pattern follows Trixi.jl: launches go through `MPI.mpiexec() do cmd; ... end`
# which sets up the launcher correctly for both system MPI and `MPI_jll`, and
# each `run(...)` is preceded by `@test true` (MPI.jl#391 workaround) so an
# mpiexec crash registers as a testset failure rather than aborting the testset.

using Test, MPI, Printf

if get(ENV, "WATERLILY_SKIP_MPI", "0") == "1"
    @info "Skipping MPI tests (WATERLILY_SKIP_MPI=1)"
    return
end

# Default rank counts. Override via env vars for resource-constrained CI.
const NP_2D   = parse(Int, get(ENV, "WATERLILY_MPI_NP_2D", "4"))
const NP_3D   = parse(Int, get(ENV, "WATERLILY_MPI_NP_3D", "8"))
const TESTDIR = @__DIR__
const WORKER  = joinpath(TESTDIR, "mpitests_workers.jl")
const OUTDIR  = mktempdir()
const JULIA   = Base.julia_cmd()
# Inherit the orchestrator's active project so workers see the test [extras]
# (MPI, ImplicitGlobalGrid, ...) without us having to configure a separate env.
const PROJECT = "--project=$(Base.active_project())"

# CI runners may have fewer hardware cores than the rank count.  Both env vars
# below tell OpenMPI / PRRTE to oversubscribe; MPICH ignores them.
const OVERSUB_ENV = ("OMPI_MCA_rmaps_base_oversubscribe" => "1",
                     "PRTE_MCA_rmaps_default_mapping_policy" => ":oversubscribe")

# Run the worker under `MPI.mpiexec()` (Trixi pattern — works with both
# system MPI and MPI_jll).  Returns true on clean exit.
function mpi_run(np::Int, role::AbstractString)
    success = false
    MPI.mpiexec() do mpiexec_cmd
        cmd = addenv(
            `$(mpiexec_cmd) -n $(np) $(JULIA) $(PROJECT) --startup-file=no
             --threads=1 --check-bounds=yes --heap-size-hint=0.5G
             $(WORKER) $(role) $(OUTDIR)`,
            OVERSUB_ENV...)
        success = Base.success(run(ignorestatus(cmd)))
    end
    return success
end

function read_toml(path)
    d = Dict{String,Float64}()
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, "#") || startswith(line, "[")) && continue
        occursin('=', line) || continue
        k, v = split(line, '=', limit=2)
        d[strip(k)] = parse(Float64, strip(v))
    end
    return d
end

# Compare two field dicts: meaningful (large-magnitude) metrics agree to `rtol`,
# near-zero symmetry components are skipped (FP32 chaos amplification — see
# `MPI_PORT.md` parity criterion in WaterLilyMeshBodies / WaterLilyIGG).
# Prints a per-metric serial/parallel/Δ table so CI logs show what was compared.
function compare_metrics(serial, parallel, name; rtol=0.02, abs_floor=0.05)
    keys_sorted = sort(collect(keys(serial)))
    println("\n  $(name) parity  (rtol=$(rtol), abs_floor=$(abs_floor))")
    @printf("    %-6s  %14s  %14s  %14s   %s\n",
            "metric", "serial", "parallel", "rel.drift", "criterion")
    @testset "$(name) parity" begin
        for k in keys_sorted
            s = serial[k]; p = parallel[k]
            drift = abs(s) > 0 ? abs(p - s) / abs(s) : abs(p - s)
            if abs(s) < abs_floor
                pass = abs(p) < max(abs_floor, 5*abs(s))
                @printf("    %-6s  %14.6g  %14.6g  %14.3g   %s\n",
                        k, s, p, drift, pass ? "≈0 ok" : "≈0 FAIL")
                @test pass
            else
                pass = abs(p - s) <= rtol * abs(s)
                @printf("    %-6s  %14.6g  %14.6g  %14.3g   %s\n",
                        k, s, p, drift, pass ? "ok" : "FAIL")
                @test pass
            end
        end
    end
end

@testset "MPI" begin
    # ── unit tests (2D grid, NP_2D ranks) ────────────────────────────────────
    @testset "unit (np=$(NP_2D))" begin
        @test true                                # MPI.jl#391 canary
        @test mpi_run(NP_2D, "unit")
    end

    # ── parity tests ──────────────────────────────────────────────────────────
    # IGG can only be initialised once per process, so cylinder and sphere
    # parallel runs each get their own subprocess.
    @testset "parity" begin
        @test true
        cmd_s = `$(JULIA) $(PROJECT) --startup-file=no $(WORKER) parity_serial $(OUTDIR)`
        @test success(run(ignorestatus(cmd_s)))

        for (case, key, np) in (("cyl", "cylinder", NP_2D),
                                ("sph", "sphere",   NP_3D))
            @testset "$(key) np=$(np)" begin
                # Remove any stale parallel result before launching: prevents a
                # crashed worker from passing the `isfile` check on leftovers.
                pfile = joinpath(OUTDIR, "parallel_$(case).toml")
                isfile(pfile) && rm(pfile)

                @test true                        # MPI.jl#391 canary
                @test mpi_run(np, "parity_parallel_$(case)")

                sfile = joinpath(OUTDIR, "serial_$(case).toml")
                @test isfile(sfile)
                @test isfile(pfile)
                if isfile(sfile) && isfile(pfile)
                    compare_metrics(read_toml(sfile), read_toml(pfile), key)
                end
            end
        end
    end
end

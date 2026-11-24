# Runs `mpitests_workers.jl` under `MPI.mpiexec()`: rank-aware unit tests, and serial/parallel parity
# of a 2D moving cylinder, a 3D sphere and a 2D anisotropic channel (multigrid depth)

using Test, MPI, Printf

if get(ENV, "WATERLILY_SKIP_MPI", "0") == "1"
    @info "Skipping MPI tests (WATERLILY_SKIP_MPI=1)"
    return
end

const NP_2D   = parse(Int, get(ENV, "WATERLILY_MPI_NP_2D", "4"))
const NP_3D   = parse(Int, get(ENV, "WATERLILY_MPI_NP_3D", "8"))
# the parity cases of every halo backend must give the native results bit for bit
const HALOS   = split(get(ENV, "WATERLILY_MPI_HALOS", "native,igg"), ',')
const TESTDIR = @__DIR__
const WORKER  = joinpath(TESTDIR, "mpitests_workers.jl")
const OUTDIR  = mktempdir()
const JULIA   = Base.julia_cmd()
const PROJECT = "--project=$(Base.active_project())"

# let OpenMPI oversubscribe the cores of CI runners
const OVERSUB_ENV = ("OMPI_MCA_rmaps_base_oversubscribe"   => "1",
                     "PRTE_MCA_rmaps_default_mapping_policy" => ":oversubscribe")

function mpi_run(np::Int, role::AbstractString; halo="native", outdir=OUTDIR)
    cmd = addenv(
        `$(MPI.mpiexec()) -n $(np) $(JULIA) $(PROJECT) --startup-file=no
         --threads=1 --check-bounds=yes --heap-size-hint=0.5G
         $(WORKER) $(role) $(outdir)`,
        OVERSUB_ENV..., "WATERLILY_HALO" => halo)
    return Base.success(run(ignorestatus(cmd)))
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

# metrics agree to `rtol`, but those below `abs_floor` (symmetry components, FP32 chaos); prints the table
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
    @testset "unit (np=$(NP_2D), halo=$(halo))" for halo in HALOS
        @test true                                # MPI.jl#391 canary
        @test mpi_run(NP_2D, "unit"; halo)
    end

    # ── parity tests ──────────────────────────────────────────────────────────
    @testset "parity" begin
        @test true
        cmd_s = `$(JULIA) $(PROJECT) --startup-file=no $(WORKER) parity_serial $(OUTDIR)`
        @test success(run(ignorestatus(cmd_s)))

        for (case, key, np) in (("cyl",  "cylinder", NP_2D),
                                ("sph",  "sphere",   NP_3D),
                                ("chan", "channel",  NP_2D))
            @testset "$(key) np=$(np)" begin
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
                @testset "halo=$(halo) ≡ native" for halo in filter(!=("native"), HALOS)
                    outdir = mkpath(joinpath(OUTDIR, halo))
                    hfile = joinpath(outdir, "parallel_$(case).toml")
                    isfile(hfile) && rm(hfile)
                    @test true                    # MPI.jl#391 canary
                    @test mpi_run(np, "parity_parallel_$(case)"; halo, outdir)
                    @test isfile(hfile) && isfile(pfile) && read_toml(hfile) == read_toml(pfile)
                end
            end
        end
    end
end

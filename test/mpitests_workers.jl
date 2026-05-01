# Child driver invoked via `mpiexec -n N julia ... mpitests_workers.jl ROLE`.
# ROLE = "unit" | "parity_serial" | "parity_parallel"
# The orchestrator (mpitests.jl) launches us, reads exit code + result files,
# and wraps everything in @testset on the parent side.

using Test
using WaterLily
using StaticArrays

const ROLE   = ARGS[1]
const OUTDIR = ARGS[2]      # writable temp dir shared with parent
const T      = Float32

# ── Parity simulations (deterministic, low-res, run identically in serial / MPI) ─

# 2D moving cylinder in a doubly-periodic domain: tests `remeasure`,
# `effective_perdir` (decomposed periodic dirs route to non-periodic stencil
# at MPI seams), and `comm!` halo exchange.
function run_cylinder!(parallel::Bool; nsteps=20)
    n, m   = 64, 32
    U      = T(1)
    radius = T(4)
    Re     = T(80)
    ν      = U * radius / Re
    center(t) = SA{T}[n/2 + 1 + T(2)*sin(T(0.3)*t), m/2 + 1]
    body = AutoBody((x, t) -> √sum(abs2, x .- center(t)) - radius)
    sim = parallel ?
        (@distributed Simulation((n, m), (U, zero(T)), radius;
            ν, body, T, perdir=(1, 2))) :
        Simulation((n, m), (U, zero(T)), radius;
            ν, body, T, perdir=(1, 2))
    sim_step!(sim; remeasure=true)  # warm-up
    for _ in 1:nsteps
        sim_step!(sim; remeasure=true)
    end
    pmax = WaterLily.global_max(maximum(abs, sim.flow.p))
    umax = WaterLily.global_max(maximum(abs, sim.flow.u))
    F    = WaterLily.total_force(sim)
    return (pmax=Float64(pmax), umax=Float64(umax),
            Fx=Float64(F[1]), Fy=Float64(F[2]),
            t=Float64(sim_time(sim)))
end

# Low-resolution 3D sphere: tests 3D halo exchange + `exitBC` under MPI.
function run_sphere!(parallel::Bool; nsteps=15)
    D = 8
    Lgrid = (6, 4, 4)
    n, m, l = Lgrid .* D                       # 48 × 32 × 32
    U      = T(1)
    radius = T(D ÷ 2)
    Re     = T(80)
    ν      = U * D / Re
    center = SA{T}[2, 2, 2] .* D
    body = AutoBody((x, t) -> √sum(abs2, x .- center) - radius)
    sim = parallel ?
        (@distributed Simulation((n, m, l), (U, zero(T), zero(T)), T(D);
            ν, body, T, exitBC=true)) :
        Simulation((n, m, l), (U, zero(T), zero(T)), T(D);
            ν, body, T, exitBC=true)
    sim_step!(sim; remeasure=false)
    for _ in 1:nsteps
        sim_step!(sim; remeasure=false)
    end
    pmax = WaterLily.global_max(maximum(abs, sim.flow.p))
    umax = WaterLily.global_max(maximum(abs, sim.flow.u))
    F    = WaterLily.total_force(sim)
    return (pmax=Float64(pmax), umax=Float64(umax),
            Fx=Float64(F[1]), Fy=Float64(F[2]), Fz=Float64(F[3]),
            t=Float64(sim_time(sim)))
end

# ── Serialise a NamedTuple to TOML so the parent can read without JLD2 ────────

function write_results(path, key, nt::NamedTuple)
    open(path, "w") do io
        println(io, "[$(key)]")
        for (k, v) in pairs(nt)
            println(io, "$(k) = $(v)")
        end
    end
end

# ── Roles ─────────────────────────────────────────────────────────────────────

if ROLE == "parity_serial"
    # IGG isn't initialised here — both cases run rank-locally back-to-back.
    cyl = run_cylinder!(false)
    sph = run_sphere!(false)
    write_results(joinpath(OUTDIR, "serial_cyl.toml"), "cylinder", cyl)
    write_results(joinpath(OUTDIR, "serial_sph.toml"), "sphere",   sph)

elseif ROLE == "parity_parallel_cyl"
    using ImplicitGlobalGrid, MPI
    cyl = run_cylinder!(true)
    WaterLily.mpi_rank() == 0 &&
        write_results(joinpath(OUTDIR, "parallel_cyl.toml"), "cylinder", cyl)
    finalize_global_grid()

elseif ROLE == "parity_parallel_sph"
    using ImplicitGlobalGrid, MPI
    sph = run_sphere!(true)
    WaterLily.mpi_rank() == 0 &&
        write_results(joinpath(OUTDIR, "parallel_sph.toml"), "sphere", sph)
    finalize_global_grid()

elseif ROLE == "unit"
    using ImplicitGlobalGrid, MPI
    using ForwardDiff: Dual, Partials, value, partials

    # Initialise a 64×32 IGG grid so the unit tests run on a real decomposition.
    local_dims, me, comm = init_waterlily_mpi((64, 32))
    np   = MPI.Comm_size(comm)
    Nloc = local_dims                         # rank-local interior dims
    ext  = Base.get_extension(WaterLily, :WaterLilyMPIExt)

    # Tests run on every rank — `Test` aborts with non-zero exit on first
    # failure, which the orchestrator surfaces via `success(run(cmd))`.
    @testset "MPI unit (rank $(me)/$(np))" begin

        @testset "par_mode + accessors" begin
            @test ext !== nothing
            @test WaterLily.par_mode[] isa ext.Parallel
            @test mpi_nprocs() == np
            @test 0 <= mpi_rank() < np
            @test mpi_comm() == comm
        end

        @testset "global reductions" begin
            # Σ rank = np*(np-1)/2
            @test WaterLily.global_allreduce(me) == np*(np-1)÷2
            @test WaterLily.global_min(T(me), T(me)) == T(0)
            @test WaterLily.global_max(T(me)) == T(np-1)
            # Σ ones (one per rank) over a length-Nloc[1] vector
            a = ones(T, Nloc[1])
            @test WaterLily.global_sum(a) ≈ T(np * Nloc[1])
            # Dot product: a⋅a = Nloc[1] per rank, np ranks
            @test WaterLily.global_dot(a, a) ≈ T(np * Nloc[1])
        end

        @testset "scalar halo exchange" begin
            # Fill rank-local interior with `me`; ghosts start at -1 sentinel.
            arr = fill(T(-1), (Nloc[1]+2, Nloc[2]+2))
            arr[2:Nloc[1]+1, 2:Nloc[2]+1] .= T(me)
            WaterLily.scalar_halo!(arr)
            g = ImplicitGlobalGrid.global_grid()
            for j in 1:2
                nleft, nright = g.neighbors[1, j], g.neighbors[2, j]
                # left ghost should hold neighbour's rank value (or -1 at physical wall)
                lslice = j == 1 ? @view(arr[1, 2:end-1]) : @view(arr[2:end-1, 1])
                rslice = j == 1 ? @view(arr[end, 2:end-1]) : @view(arr[2:end-1, end])
                if nleft >= 0
                    @test all(lslice .== T(nleft))
                else
                    @test all(lslice .== T(-1))
                end
                if nright >= 0
                    @test all(rslice .== T(nright))
                else
                    @test all(rslice .== T(-1))
                end
            end
        end

        @testset "global_offset + @loop coords" begin
            # Inside @loop, `loc(0,I)` returns global coords.  Build coord array
            # via `apply!` (which uses @loop) and compare to expected globals.
            arr = zeros(T, Nloc[1]+2, Nloc[2]+2)
            apply!((i,x) -> i==1 ? x[1] : zero(T), reshape(arr, size(arr)..., 1))
            offset = WaterLily.global_offset(Val(2), T)
            # First interior cell in x: local index 2, face 1 → loc = 0 + offset[1]
            @test arr[2, 2] ≈ offset[1]
            @test arr[3, 2] ≈ offset[1] + 1
        end

        @testset "phys_left / phys_right gates" begin
            g = ImplicitGlobalGrid.global_grid()
            for j in 1:2
                @test WaterLily.phys_left(j)  == (g.neighbors[1, j] < 0)
                @test WaterLily.phys_right(j) == (g.neighbors[2, j] < 0)
            end
        end

        @testset "decomposed + effective_perdir" begin
            g = ImplicitGlobalGrid.global_grid()
            for j in 1:2
                @test WaterLily.decomposed(j) == (g.dims[j] > 1)
            end
            @test WaterLily.effective_perdir((1, 2)) ==
                  Tuple(j for j in (1, 2) if !WaterLily.decomposed(j))
        end

        @testset "shape-aware topology" begin
            # Known asymmetric case (1024×64, 8 ranks → (8,1) wins on halo
            # surface vs (4,2)); isotropic case where Dims_create's tiebreak wins.
            @test ext._shape_aware_topology((1024, 64), 8) == (8, 1)
            @test ext._shape_aware_topology((128, 128), 4) == (2, 2)
        end

        @testset "ForwardDiff.Dual reductions" begin
            # `Dual{Tag,V,N}` reductions reinterpret to flat V[value, partials...].
            # Per-rank value (me+1, partials=(1, me)) — chosen so SUM/MIN/MAX
            # produce distinct global answers in both the value and partials.
            mkdual2(v, p1, p2) = Dual{Nothing,Float64,2}(v, Partials{2,Float64}((p1, p2)))
            mkdual1(v, p1)     = Dual{Nothing,Float64,1}(v, Partials{1,Float64}((p1,)))
            x = mkdual2(Float64(me + 1), 1.0, Float64(me))

            # SUM: value Σ(me+1) = np(np+1)/2;  partials Σ(1, me) = (np, np(np-1)/2)
            s = WaterLily.global_allreduce(x)
            @test value(s) ≈ np * (np + 1) / 2
            @test partials(s)[1] ≈ np
            @test partials(s)[2] ≈ np * (np - 1) / 2

            # MIN: lowest value lives on rank 0 → me=0 partials (1, 0)
            mn = WaterLily.global_min(x, x)
            @test value(mn) ≈ 1.0
            @test partials(mn)[1] ≈ 1.0
            @test partials(mn)[2] ≈ 0.0

            # MAX: highest value lives on rank np-1 → partials (1, np-1)
            mx = WaterLily.global_max(x)
            @test value(mx) ≈ Float64(np)
            @test partials(mx)[1] ≈ 1.0
            @test partials(mx)[2] ≈ Float64(np - 1)

            # Array SUM (in-place via flat-V reinterpret).
            a = fill(mkdual2(Float64(me), Float64(me), 1.0), 3)
            sa = WaterLily.global_allreduce(a)
            sumvals = sum(0:np-1)               # Σ rank
            for k in 1:length(a)
                @test value(sa[k])       ≈ sumvals
                @test partials(sa[k])[1] ≈ sumvals
                @test partials(sa[k])[2] ≈ Float64(np)
            end

            # Halo exchange of a Dual array routes through `_scalar_halo_mpi!`
            # (Dual eltype is non-native — IGG's typed buffers can't carry it).
            arr = fill(mkdual1(-1.0, -1.0), (Nloc[1]+2, Nloc[2]+2))
            interior = mkdual1(Float64(me), Float64(me))
            arr[2:Nloc[1]+1, 2:Nloc[2]+1] .= interior
            WaterLily.scalar_halo!(arr)
            g = ImplicitGlobalGrid.global_grid()
            for j in 1:2
                nleft, nright = g.neighbors[1, j], g.neighbors[2, j]
                lslice = j == 1 ? @view(arr[1, 2:end-1]) : @view(arr[2:end-1, 1])
                rslice = j == 1 ? @view(arr[end, 2:end-1]) : @view(arr[2:end-1, end])
                if nleft >= 0
                    @test all(value(d)       ≈ Float64(nleft) for d in lslice)
                    @test all(partials(d)[1] ≈ Float64(nleft) for d in lslice)
                end
                if nright >= 0
                    @test all(value(d)       ≈ Float64(nright) for d in rslice)
                    @test all(partials(d)[1] ≈ Float64(nright) for d in rslice)
                end
            end
        end
    end

    finalize_global_grid()

else
    error("Unknown ROLE: $(ROLE)")
end

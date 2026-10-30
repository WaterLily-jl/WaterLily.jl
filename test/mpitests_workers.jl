# Child driver invoked via `mpiexec -n N julia ... mpitests_workers.jl ROLE`.
# ROLE = "unit" | "parity_serial" | "parity_parallel"
# The orchestrator (mpitests.jl) launches us, reads exit code + result files,
# and wraps everything in @testset on the parent side.

using Test
using WaterLily
using WaterLily: @loop, loc, inside, δ
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

# Anisotropic 2D channel (aspect 4, Neumann pressure on every wall), decomposed along the
# long axis: guards the multigrid depth under MPI. With too shallow a hierarchy the lowest
# Neumann mode of the long direction survives the residual-based stopping criterion and
# the pressure (not the velocity) drifts from serial by tens of percent.
function run_channel!(parallel::Bool; nsteps=20)
    n, m   = 256, 64
    U      = T(1)
    radius = T(8)
    Re     = T(100)
    ν      = U * radius / Re
    center = SA{T}[n/4, m/2]
    body = AutoBody((x, t) -> √sum(abs2, x .- center) - radius)
    sim = parallel ?
        (@distributed Simulation((n, m), (U, zero(T)), radius; ν, body, T)) :
        Simulation((n, m), (U, zero(T)), radius; ν, body, T)
    sim_step!(sim; remeasure=false)
    for _ in 1:nsteps
        sim_step!(sim; remeasure=false)
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

# A body with a `bbox`: another body's fields, measured only over a fixed box of global
# cell indices. The box isn't intersected with `R`, which is fine in serial since it lies
# in the domain; under MPI the rank's share of it has to come from WaterLily.
struct Boxed{B} <: WaterLily.AbstractBody
    body::B
    box::CartesianIndices{2,NTuple{2,UnitRange{Int}}}
end
WaterLily.measure(b::Boxed, x, t; kwargs...) = WaterLily.measure(b.body, x, t; kwargs...)
WaterLily.sdf(b::Boxed, x, t=0; kwargs...) = WaterLily.sdf(b.body, x, t; kwargs...)
WaterLily.bbox(b::Boxed, R) = b.box

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
    # MPI isn't initialised here — the cases run rank-locally back-to-back.
    cyl = run_cylinder!(false)
    sph = run_sphere!(false)
    chan = run_channel!(false)
    write_results(joinpath(OUTDIR, "serial_cyl.toml"),  "cylinder", cyl)
    write_results(joinpath(OUTDIR, "serial_sph.toml"),  "sphere",   sph)
    write_results(joinpath(OUTDIR, "serial_chan.toml"), "channel",  chan)

elseif ROLE == "parity_parallel_cyl"
    using MPI
    cyl = run_cylinder!(true)
    WaterLily.mpi_rank() == 0 &&
        write_results(joinpath(OUTDIR, "parallel_cyl.toml"), "cylinder", cyl)

elseif ROLE == "parity_parallel_sph"
    using MPI
    sph = run_sphere!(true)
    WaterLily.mpi_rank() == 0 &&
        write_results(joinpath(OUTDIR, "parallel_sph.toml"), "sphere", sph)

elseif ROLE == "parity_parallel_chan"
    using MPI
    chan = run_channel!(true)
    WaterLily.mpi_rank() == 0 &&
        write_results(joinpath(OUTDIR, "parallel_chan.toml"), "channel", chan)

elseif ROLE == "unit"
    using MPI
    using ForwardDiff: Dual, Partials, value, partials

    # Initialise a 64×32 grid so the unit tests run on a real decomposition.
    local_dims, me, comm = init_waterlily_mpi((64, 32))
    np   = MPI.Comm_size(comm)
    Nloc = local_dims                         # rank-local interior dims
    ext  = Base.get_extension(WaterLily, :WaterLilyMPIExt)

    # Every ghost must hold the value of the cell it stands for: its global index (G₁ + 1000G₂ + 10⁶G₃, plus
    # 10⁹ per component), -1 past a physical boundary. Fine, coarse and velocity-shaped arrays; Dual and Bool too.
    function halo_selfcheck(ldims, perdir)
        p, D = ext._par(), length(ldims)
        for sz in (Tuple(ldims .+ 2), Tuple(ldims .÷ 2 .+ 2), (Tuple(ldims .+ 2)..., D))
            n = sz[1:D] .- 2
            function cell(I)
                G = ntuple(k -> I[k] - 1 + p.coords[k]*n[k], D)
                G = ntuple(k -> p.periods[k] ? mod1(G[k], p.dims[k]*n[k]) : G[k], D)
                all(k -> 1 <= G[k] <= p.dims[k]*n[k], 1:D) ? sum(G[k]*1000^(k-1) for k in 1:D) + 10^9*get(Tuple(I), D+1, 1) : -1
            end
            ghost(I) = any(k -> I[k] in (1, sz[k]), 1:D)
            want = [Float64(cell(I)) for I in CartesianIndices(sz)]
            a0 = [ghost(I) ? -1.0 : want[I] for I in CartesianIndices(sz)]
            a, d = copy(a0), Dual{Nothing,Float64,1}.(a0, Partials.(tuple.(-a0)))
            b = [!ghost(I) && isodd(Int(want[I])) for I in CartesianIndices(sz)]
            comm!(x) = length(sz) > D ? WaterLily.velocity_comm!(x, perdir) : WaterLily.comm!(x, perdir)
            foreach(comm!, (a, d, b))
            @test a == want && a != a0
            @test value.(d) == want && partials.(d, 1) == -want
            @test b == [w != -1 && isodd(Int(w)) for w in want]
        end
    end

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

        @testset "loc in @loop bodies" begin
            off = global_offset(Val(2), Float64)
            a = zeros(Nloc .+ 2)
            @loop a[I] = loc(0,I,Float64)[1] over I ∈ inside(a)           # global coordinates
            @test all(a[I] == loc(0,I,Float64)[1] + off[1] for I in inside(a))
            @loop a[I] = WaterLily.loc(0,I,Float64)[2] over I ∈ inside(a) # qualified name: local
            @test all(a[I] == loc(0,I,Float64)[2] for I in inside(a))
            let loc = fill(3.0, size(a))                                    # a variable named loc
                @loop a[I] = loc[I] over I ∈ inside(a)
            end
            @test all(a[I] == 3 for I in inside(a))
            off32 = global_offset(Val(2), Float32)                          # not float or Dual: Float32 offset
            v = fill(zero(SVector{2,Float64}), size(a))
            @loop v[I] = loc(0,I) over I ∈ inside(v)
            @test all(v[I] == loc(0,I) + off32 for I in inside(v))
            n = zeros(Int, size(a))
            @loop n[I] = floor(Int, loc(0,I)[1]) over I ∈ inside(n)
            @test all(n[I] == floor(Int, loc(0,I)[1] + off32[1]) for I in inside(n))
            b = zeros(Bool, size(a))
            @loop b[I] = loc(0,I)[1] > 40 over I ∈ inside(b)
            @test all(b[I] == (loc(0,I)[1] + off32[1] > 40) for I in inside(b))
        end

        @testset "topology accessors" begin
            p = ext._par()
            @test p === WaterLily.par_mode[] && ext._nd() == 2 && prod(p.dims) == np
            @test ext._any_decomposed() == any(>(1), p.dims)
            @test p.coords == MPI.Cart_coords(comm) && p.nlocal == collect(Nloc)
            @test all(p.neighbors[:, j] == collect(MPI.Cart_shift(comm, j-1, 1)) for j in 1:2)
        end

        @testset "global reductions" begin
            # Σ rank = np*(np-1)/2
            @test WaterLily.global_allreduce(me) == np*(np-1)÷2
            v = fill(T(me), 3); w = WaterLily.global_allreduce(v) # a new array, v unchanged
            @test w == fill(T(np*(np-1)÷2), 3) && v == fill(T(me), 3)
            @test WaterLily.global_allreduce!(v) === v && v == w      # in place
            @test WaterLily.global_min(T(me), T(me)) == T(0)
            @test WaterLily.global_max(T(me)) == T(np-1)
            @test WaterLily.global_sum_max(T(me), T(2me)) === (T(np*(np-1)÷2), T(2np-2)) # one Allgather
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
            g = ext._par()
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

        @testset "halo: every ghost holds the value of its cell, edges and corners too" begin
            halo_selfcheck(Nloc, ())
        end

        @testset "inner + shell: every cell of a range once" begin
            # inner(R) reads no ghost of a decomposed dim (both are, 2×2 ranks);
            # with shell(R) it covers R once, also when R is one or two cells thick
            for R in (inside(zeros(T, Nloc .+ 2)), CartesianIndices((2:2, 2:5)), CartesianIndices((2:3, 4:4)))
                I, S = vec(collect(WaterLily.inner(R))), collect(WaterLily.shell(R))
                @test sort!(vcat(I, S)) == sort!(vec(collect(R)))
                @test all(J -> all(j -> !WaterLily.decomposed(j) || first(R.indices[j]) < J[j] < last(R.indices[j]), 1:2), I)
            end
        end

        @testset "comm!(f, ...): the loop split around the halo ≡ after it" begin
            x = rand(T, Nloc .+ 2); y = copy(x); r1, r2 = zeros(T, size(x)), zeros(T, size(x))
            ext._overlap_min[] = 0  # split every loop
            WaterLily.comm!(x, ()) do R
                @loop r1[I] = x[I-δ(1,I)]+x[I+δ(1,I)]+x[I-δ(2,I)]+x[I+δ(2,I)] over I ∈ R
            end
            ext._overlap_min[] = 2^20
            WaterLily.comm!(y, ())
            @loop r2[I] = y[I-δ(1,I)]+y[I+δ(1,I)]+y[I-δ(2,I)]+y[I+δ(2,I)] over I ∈ inside(y)
            @test x == y && r1 == r2
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

        @testset "bbox in global cell indices" begin
            # Circles boxed a few cells wider than the band `measure!` fills: the first across
            # the x and y seams, the second left of x = 33, so the ranks right of it loop over nothing.
            r, R = T(6), inside(zeros(T, Nloc .+ 2))
            for (c, box) in ((SA{T}[33, 17], (24:45, 8:29)), (SA{T}[20, 17], (11:32, 8:29)))
                circle = AutoBody((x, t) -> √sum(abs2, x .- c) - r)
                boxed = Boxed(circle, CartesianIndices(box))
                Rb = WaterLily.local_bbox(boxed, R)
                @test Rb ⊆ R
                @test WaterLily.global_allreduce(length(Rb)) == length(boxed.box) # the blocks tile the box
                c[1] < 33 && ext._par().dims[1] > 1 &&
                    @test WaterLily.global_allreduce(Int(isempty(Rb))) > 0
                a, b = (Simulation(Nloc, (T(1), T(0)), 2r; body, T) for body in (circle, boxed))
                foreach(sim_step!, (a, b))
                @test a.flow.μ₀ == b.flow.μ₀ && a.flow.μ₁ == b.flow.μ₁ && a.flow.V == b.flow.V
                @test a.flow.p == b.flow.p && a.flow.u == b.flow.u
                @test WaterLily.total_force(a) == WaterLily.total_force(b)
                @test WaterLily.viscous_force(a) == WaterLily.viscous_force(b)
                @test WaterLily.pressure_moment(c, a) == WaterLily.pressure_moment(c, b)
            end
        end

        @testset "phys_left / phys_right gates" begin
            g = ext._par()
            for j in 1:2
                @test WaterLily.phys_left(j)  == (g.neighbors[1, j] < 0)
                @test WaterLily.phys_right(j) == (g.neighbors[2, j] < 0)
            end
        end

        @testset "decomposed + effective_perdir" begin
            g = ext._par()
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

            # SUM and MAX together: the two reductions above
            sm = WaterLily.global_sum_max(x, x)
            @test sm[1] == s && sm[2] == mx

            # Array SUM (via the flat-V alias), into a copy or in place.
            a = fill(mkdual2(Float64(me), Float64(me), 1.0), 3)
            sa = WaterLily.global_allreduce(a)
            @test all(==(mkdual2(Float64(me), Float64(me), 1.0)), a)
            @test WaterLily.global_allreduce!(a) === a && a == sa
            sumvals = sum(0:np-1)               # Σ rank
            for k in 1:length(a)
                @test value(sa[k])       ≈ sumvals
                @test partials(sa[k])[1] ≈ sumvals
                @test partials(sa[k])[2] ≈ Float64(np)
            end

            # Halo exchange of a Dual array (MPI sends it as a derived datatype).
            arr = fill(mkdual1(-1.0, -1.0), (Nloc[1]+2, Nloc[2]+2))
            interior = mkdual1(Float64(me), Float64(me))
            arr[2:Nloc[1]+1, 2:Nloc[2]+1] .= interior
            WaterLily.scalar_halo!(arr)
            g = ext._par()
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

            # Velocity-shaped Dual array: all components at once.
            u = fill(mkdual1(-1.0, -1.0), (Nloc[1]+2, Nloc[2]+2, 2))
            for c in 1:2
                u[2:Nloc[1]+1, 2:Nloc[2]+1, c] .= mkdual1(Float64(me + 10c), Float64(c))
            end
            WaterLily.velocity_halo!(u)
            for c in 1:2, j in 1:2
                nleft, nright = g.neighbors[1, j], g.neighbors[2, j]
                lslice = j == 1 ? @view(u[1, 2:end-1, c]) : @view(u[2:end-1, 1, c])
                rslice = j == 1 ? @view(u[end, 2:end-1, c]) : @view(u[2:end-1, end, c])
                if nleft >= 0
                    @test all(value(d)       ≈ Float64(nleft + 10c) for d in lslice)
                    @test all(partials(d)[1] ≈ Float64(c)           for d in lslice)
                end
                if nright >= 0
                    @test all(value(d)       ≈ Float64(nright + 10c) for d in rslice)
                    @test all(partials(d)[1] ≈ Float64(c)            for d in rslice)
                end
            end
        end

        @testset "halo, 3D and periodic: every ghost holds the value of its cell" begin
            # last, since it changes the decomposition: 2×2×1 ranks, so perBC! wraps z and the halo x and y
            ld, _, _ = init_waterlily_mpi((16, 16, 8); perdir=(1, 2, 3))
            halo_selfcheck(ld, (1, 2, 3))
        end
    end


else
    error("Unknown ROLE: $(ROLE)")
end

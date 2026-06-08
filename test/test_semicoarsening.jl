# Semi-coarsening multigrid tests.
#
# Standard geometric multigrid coarsens every direction 2× per level and stops once the
# SMALLEST direction bottoms out (`divisible(N) = mod(N,2)==0 && N>4`). On non-cubic
# grids that leaves the large directions coarse-grained too little, so low-frequency
# error is never corrected and the V-cycle stalls. Semi-coarsening instead coarsens each
# direction independently (only the still-`divisible` ones), so the level count is set by
# the LARGEST direction (up to `maxlevels`) rather than the smallest.
#
# Run standalone:  julia --project=. test/test_semicoarsening.jl
# (also included by maintests.jl)

using WaterLily, Test, StaticArrays
using WaterLily: L₂, inside, MultiLevelPoisson, Poisson, coarsen_mask, divisible

# Solve A x = z for a known linear field and report relative solution error + the solver.
function mlpois_setup(N::NTuple{D}; T=Float32, f=Array) where D
    c = ones(T,N...,D) |> f; BC!(c, ntuple(zero,D))
    x = zeros(T,N) |> f; z = copy(x)
    pois = MultiLevelPoisson(x,c,z)
    soln = map(I->T(I.I[1]), CartesianIndices(N)) |> f
    I = first(inside(x)); soln .-= soln[I]
    z .= mult!(pois, soln)
    solver!(pois)
    x .-= x[I]
    return L₂(x-soln)/L₂(soln), pois
end

@testset "semi-coarsening masks & operators" begin
    # coarsen only directions that are still divisible (even and >4)
    @test coarsen_mask((18,18,6)) == (true,true,true)
    @test coarsen_mask((18,18,4)) == (true,true,false) # 4 is not >4
    @test coarsen_mask((18,17,6)) == (true,false,true) # 17 is odd
    @test coarsen_mask((4,4))     == (false,false)     # nothing left to coarsen

    # masked up/down: flagged dims coarsen 2×, the rest are frozen (identity)
    c = (true,true,false); I = CartesianIndex(4,3,5)
    @test all(WaterLily.down(J,c)==I for J ∈ WaterLily.up(I,c)) # down∘up identity
    @test all(J.I[3]==I.I[3] for J ∈ WaterLily.up(I,c))         # frozen direction
    @test length(WaterLily.up(I,c)) == 4                        # 2×2×1 fine cells
    # the all-true mask reproduces full coarsening exactly
    @test collect(WaterLily.up(I,(true,true,true))) == collect(WaterLily.up(I))
    @test all(WaterLily.down(J,(true,true,true))==WaterLily.down(J) for J ∈ WaterLily.up(I))
end

@testset "semi-coarsening: cubic grids unchanged" begin
    # equal dimensions all hit the coarsening floor together ⇒ identical to full coarsening
    err,p = mlpois_setup((2^6+2,2^6+2))
    @test length(p.levels)==6 && p.n[]≤3 && err<1e-6
    err,p = mlpois_setup((2^4+2,2^4+2,2^4+2))
    @test length(p.levels)==4 && p.n[]≤3 && err<1e-6
end

@testset "semi-coarsening: rectangular grids converge" begin
    # On these power-of-2 but non-cubic grids, full coarsening stalls (the small direction
    # caps the level count): 258×34 hit the iteration cap without reaching tol, and
    # 258×66×18 only reached err≈0.3. Semi-coarsening keeps coarsening the large directions
    # to a small coarsest grid and converges normally.
    println("  semi-coarsening convergence (N → levels, n, err):")
    for (N, maxlvl, maxn, maxerr) in (((2^7+2,2^5+2), 7, 8,  1e-5),   # 130×34  (4:1)
                                      ((2^8+2,2^5+2), 8, 15, 1e-4),   # 258×34  (8:1)
                                      ((2^8+2,2^6+2,2^4+2), 7, 32, 1e-4)) # 258×66×18
        err,p = mlpois_setup(N)
        println("    $N → $(length(p.levels)), $(p.n[]), $(round(err,sigdigits=3))  coarsest=$(size(p.levels[end].x))")
        @test length(p.levels) ≥ maxlvl     # large direction keeps coarsening
        @test all(size(p.levels[end].x) .≤ 6)  # coarsest grid is genuinely coarse in every dim
        @test p.n[] ≤ maxn
        @test err < maxerr
    end
end

@testset "semi-coarsening: thin (high aspect-ratio) grids" begin
    # 66×66×6: full coarsening bottoms out at the small z after a single step (2 levels) and
    # would throw "requires size=a2ⁿ"; semi-coarsening builds a full hierarchy and solves.
    err,p = mlpois_setup((2^6+2,2^6+2,2^2+2))
    @test length(p.levels) ≥ 5
    @test size(p.levels[end].x,3) == 4   # z frozen once it bottoms out
    @test size(p.levels[end].x,1) ≤ 6    # x kept coarsening to the coarsest grid
    @test err < 1e-4
    # a thicker thin-domain converges in few iterations
    err,p = mlpois_setup((2^6+2,2^6+2,2^4+2))   # 66×66×18
    @test p.n[] ≤ 8 && err < 1e-6
end

function Poisson_setup(poisson,N::NTuple{D};f=Array,T=Float32) where D
    c = ones(T,N...,D) |> f; BC!(c, ntuple(zero,D))
    x = zeros(T,N) |> f; z = copy(x)
    pois = poisson(x,c,z)
    soln = map(I->T(I.I[1]),CartesianIndices(N)) |> f
    I = first(inside(x))
    GPUArrays.@allowscalar @. soln -= soln[I]
    z = mult!(pois,soln)
    solver!(pois)
    GPUArrays.@allowscalar @. x -= x[I]
    return L₂(x-soln)/L₂(soln),pois
end

@testset "Poisson.jl" begin
    for f ∈ arrays
        err,pois = Poisson_setup(Poisson,(5,5);f)
        @test GPUArrays.@allowscalar parent(pois.D)==f(Float32[0 0 0 0 0; 0 -2 -3 -2 0; 0 -3 -4 -3 0;  0 -2 -3 -2 0; 0 0 0 0 0])
        @test GPUArrays.@allowscalar parent(pois.iD)≈f(Float32[0 0 0 0 0; 0 -1/2 -1/3 -1/2 0; 0 -1/3 -1/4 -1/3 0;  0 -1/2 -1/3 -1/2 0; 0 0 0 0 0])
        @test err < 1e-5
        err,pois = Poisson_setup(Poisson,(2^6+2,2^6+2);f)
        @test err < 5e-6          # looser solve at default tol=2e-3 (was <1e-6 at the old tighter default)
        @test pois.n[] < 340
        @test WaterLily.L∞(pois) < 2e-3   # max-norm (L∞) tolerance is met
        err,pois = Poisson_setup(Poisson,(2^4+2,2^4+2,2^4+2);f)
        @test err < 1e-6
        @test pois.n[] < 40
    end
    for f ∈ arrays
        Ng = (8,8,8)
        σ1 = rand(Ng...) |> f
        σ2 = rand(Ng...) |> f
        @test GPUArrays.@allowscalar WaterLily.perdot(σ1,σ2,())    ≈ sum(σ1[I]*σ2[I] for I∈CartesianIndices(σ1))
        @test GPUArrays.@allowscalar WaterLily.perdot(σ1,σ2,(1,))  ≈ sum(σ1[I]*σ2[I] for I∈inside(σ1))
        @test GPUArrays.@allowscalar WaterLily.perdot(σ1,σ2,(1,2)) ≈ sum(σ1[I]*σ2[I] for I∈inside(σ1))
    end
end

@testset "MultiLevelPoisson.jl" begin
    # full-coarsening up/down
    I = CartesianIndex(4,3,2)
    @test all(WaterLily.down(J)==I for J ∈ WaterLily.up(I))
    @test_throws AssertionError("MultiLevelPoisson requires size=a2ⁿ, where n>2") Poisson_setup(MultiLevelPoisson,(15+2,3^4+2))

    # semi-coarsening masks: coarsen only divisible (even and >4) directions
    @test WaterLily.coarsen_mask((18,18,6)) == (true,true,true)
    @test WaterLily.coarsen_mask((18,18,4)) == (true,true,false) # 4 is not >4
    @test WaterLily.coarsen_mask((18,17,6)) == (true,false,true) # 17 is odd
    # masked up/down: flagged dims coarsen 2×, rest frozen
    c = (true,true,false); I = CartesianIndex(4,3,5)
    @test all(WaterLily.down(J,c)==I for J ∈ WaterLily.up(I,c))  # down∘up identity
    @test all(J.I[3]==I.I[3] for J ∈ WaterLily.up(I,c))          # frozen direction
    @test collect(WaterLily.up(I,(true,true,true))) == collect(WaterLily.up(I)) # all-true = full coarsening

    err,pois = Poisson_setup(MultiLevelPoisson,(10,10))
    @test pois.levels[3].D == Float32[0 0 0 0; 0 -2 -2 0; 0 -2 -2 0; 0 0 0 0]
    @test err < 1e-5

    pois.levels[1].L[5:6,:,1].=0
    WaterLily.update!(pois)
    @test pois.levels[3].D == Float32[0 0 0 0; 0 -1 -1 0; 0 -1 -1 0; 0 0 0 0]

    for f ∈ arrays
        err,pois = Poisson_setup(MultiLevelPoisson,(2^6+2,2^6+2);f)
        @test err < 1e-6
        @test pois.n[] ≤ 4
        @test WaterLily.L∞(pois.levels[1]) < 2e-3   # max-norm (L∞) tolerance is met at default tol=2e-3
        err,pois = Poisson_setup(MultiLevelPoisson,(2^4+2,2^4+2,2^4+2);f)
        @test err < 1e-6
        @test pois.n[] ≤ 3
    end

    # semi-coarsening check
    let H=2^4,R=H÷4,ctr=SA[4H,H÷2]        # 2D 8:1 channel with 50% blocking circle
        sim=Simulation((8H,H),(1,0),R;ν=R/100,body=AutoBody((x,t)->√sum(abs2,x-ctr)-R),T=Float32)
        foreach(_->sim_step!(sim;remeasure=false),1:4)
        @test all(sim.pois.n .≤ 10) && !any(isnan.(sim.pois.n))
    end
    let H=2^3,R=H÷4,ctr=SA[4H,H÷2,H÷2]   # 3D 8:1:1 duct with 50% blocking sphere
        sim=Simulation((8H,H,H),(1,0,0),R;ν=R/100,body=AutoBody((x,t)->√sum(abs2,x-ctr)-R),T=Float32)
        foreach(_->sim_step!(sim;remeasure=false),1:4)
        @test all(sim.pois.n .≤ 12) && !any(isnan.(sim.pois.n))
    end
end

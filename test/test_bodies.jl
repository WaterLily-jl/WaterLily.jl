@testset "Body.jl" begin
    @test WaterLily.μ₀(3.,6)==WaterLily.μ₀(0.5,1)
    @test WaterLily.μ₀(0.,1)==0.5
    @test WaterLily.μ₀(eps(1.)-1,1)==0
    @test WaterLily.μ₁(0.,2)==2*(1/4-1/π^2)

    @test all(measure(WaterLily.NoBody(),[2,1],0) .== (Inf,zeros(2),zeros(2)))
    @test sdf(WaterLily.NoBody(),[2,1],0) == Inf
end

@testset "AutoBody.jl" begin
    # test AutoDiff in 2D and 3D
    circ(x,t)=√sum(abs2,x)-2
    body1 = AutoBody((x,t)->circ(x,t)-t)
    body2 = AutoBody(circ,(x,t)->x.+t^2)
    @test all(measure(body1,[√2.,√2.],0.).≈(0,[√.5,√.5],[0.,0.]))
    @test all(measure(body1,[2.,0.,0.],1.).≈(-1.,[1.,0.,0.],[0.,0.,0.]))
    @test all(measure(body2,[√2.,√2.],0.).≈(0,[√.5,√.5],[0.,0.]))
    @test all(measure(body2,[1.,-1.,-1.],1.).≈(0.,[1.,0.,0.],[-2.,-2.,-2.]))

    # test booleans
    @test all(measure(body1+body2,[-√2.,-√2.],1.).≈(-√2.,[-√.5,-√.5],[-2.,-2.]))
    @test all(measure(body1∪body2,[-√2.,-√2.],1.).≈(-√2.,[-√.5,-√.5],[-2.,-2.]))
    @test all(measure(body1-body2,[-√2.,-√2.],1.).≈(√2.,[√.5,√.5],[-2.,-2.]))

    # test sdf and exactly equal distance bodies
    @test sdf(AutoBody(circ)+AutoBody(circ,(x,t)->x.-[6,0]),[3.,0.],0.) == 1

    # test scaling
    body = AutoBody(circ)
    for i in 2:20
        body += AutoBody(circ,(x,t)->x-rand(2))
        @test sizeof(body) ≤ i
    end

    # test curvature, 2D and 3D
    # A = ForwardDiff.Hessian(y->body1.sdf(y,0.0),[0.,0.])
    @test all(WaterLily.curvature([1. 0.; 0. 1.]).≈(1.,0.))
    @test all(WaterLily.curvature([2. 1. 0.; 1. 2. 1.; 0. 1. 2.]).≈(3.,10.))

    # check sdf on arrays and that it recovers set arithmetic identity
    for f ∈ arrays
        p = zeros(Float32,4,5) |> f; measure_sdf!(p,(body1 ∩ body2) ∪ body1)
        for I ∈ inside(p)
            @test GPUArrays.@allowscalar p[I]≈sdf(body1,loc(0,I,Float32))
        end
    end

    # check fast version
    @test all(measure(body1,[3.,4.],0.,fastd²=9) .≈ measure(body1,[3.,4.],0.))
    @test all(measure(body1,[3.,4.],0.,fastd²=8) .≈ (sdf(body1,[3.,4.],0.,fastd²=9),zeros(2),zeros(2)))
end

@testset "RigidMap.jl" begin
    for T ∈ (Float32,Float64)
        # initialize a rigid body
        sdf(x,t) = sqrt(sum(abs2,x))-1
        body = AutoBody(sdf, RigidMap(SA{T}[0,0],T(0)))
        # check sdf
        @test all(measure(body,SA{T}[1.5,0],0) .≈ (1/2,SA{T}[1,0],SA{T}[0,0]))
        # rotate and add linear velocity
        body = setmap(body;θ=T(π/4),V=SA{T}[1.0,0])
        # check sdf and velocity
        @test all(measure(body,SA{T}[1.5,0],0) .≈ (1/2,SA{T}[1,0],SA{T}[1,0]))
        # add angular velocity
        body = setmap(body;ω=T(0.1))
        @test all(measure(body,SA{T}[1.5,0],0) .≈ (1/2,SA{T}[1,0],SA{T}[1,1.5*0.1]))
        # 3D rigid body
        body3D = AutoBody(sdf, RigidMap(SA{T}[0,0,0],SA{T}[0,0,0];xₚ=SA{T}[-.5,0,0]))
        @test all(measure(body3D,SA{T}[1.5,0,0],0) .≈ (1/2,SA{T}[1,0,0],SA{T}[0,0,0]))
        # test rotations about x, y, and z
        # rotate by 180 degrees about x-axis, should not change
        body3D = setmap(body3D;θ=SA{T}[π,0,0])
        @test all(measure(body3D,SA{T}[1.5,0,0],0) .≈ (1/2,SA{T}[1,0,0],SA{T}[0,0,0]))
        # now rotate by 180 around y=axis, should invert z-component of normal
        body3D = setmap(body3D;θ=SA{T}[0,π,0],V=SA{T}[1.0,0,0])
        @test all(measure(body3D,SA{T}[1.5,0,0],0) .≈ (1.5,SA{T}[1,0,0],SA{T}[1,0,0]))
        body3D = setmap(body3D;θ=SA{T}[0,0,π],V=SA{T}[1.0,0,0])
        @test all(measure(body3D,SA{T}[1.5,0,0],0) .≈ (1.5,SA{T}[1,0,0],SA{T}[1,0,0]))
        # 3D rigid body with linear and angular velocity
        body3D = setmap(body3D;θ=SA{T}[0,0,0],V=SA{T}[1.0,0,0],ω=SA{T}[0,0,0.1])
        @test all(measure(body3D,SA{T}[1.5,0,0],0) .≈ (1/2,SA{T}[1,0,0],SA{T}[1,0.2,0]))
        @test all(measure(body3D,SA{T}[0,1.5,0],0) .≈ (1/2,SA{T}[0,1,0],SA{T}[0.85,0.05,0]))
        @test all(measure(body3D,SA{T}[1.5,1.5,1.5],0) .≈ (√(3*(1.5^2))-1,SA{T}[√(1/3),√(1/3),√(1/3)],SA{T}[.85,0.2,0]))
        # three 3D rotations
        body3D = setmap(body3D;V=SA{T}[1.0,0,0],ω=SA{T}[0,-0.1,0.1])
        @test all(measure(body3D,SA{T}[1.5,0,0],0) .≈ (1/2,SA{T}[1,0,0],SA{T}[1,0.2,0.2]))
        @test all(measure(body3D,SA{T}[0,1.5,1.5],0) .≈ (√(2*(1.5^2))-1,SA{T}[0,√(1/2),√(1/2)],SA{T}[0.7,0.05,0.05]))
        # test for a SetMap
        body = AutoBody(sdf, RigidMap(SA{T}[0,0],T(0))) +AutoBody(sdf, RigidMap(SA{T}[1,1],T(0)))
        body = setmap(body;θ=T(π/4),V=SA{T}[1.0,0])
        @test all(body.a.map.θ == body.b.map.θ  == T(π/4))
        @test all(body.a.map.V .≈ body.b.map.V  .≈ [1,0])
        # try measure in the sim using different backends
        for array in arrays
            body = AutoBody((x,t)->sqrt(sum(abs2,x))-4,RigidMap(SA{T}[16,16,16],SA{T}[0,0,0];
                             V=SA{T}[0,0,0],ω=SA{T}[0,-0.1,0.1]))
            sim = Simulation((32,32,32),(1,0,0),8;body,T,mem=array)
            @test GPUArrays.@allowscalar all(extrema(sim.flow.V) .≈ (-0.9,0.9))
            sim.body = setmap(sim.body;x₀=SA{T}[16,16,12])
            @test GPUArrays.@allowscalar all(sim.flow.μ₀[17,17,17,:] .≈ 0)
        end
    end
    rmap = RigidMap(SA[0.,0.],π/4)
    body = AutoBody((x,t)->√(x'x)-1,rmap)-AutoBody((x,t)->√(x'x)-0.5,rmap) # annulus
    @test all(measure(setmap(body,ω=1.),SA[0.25,0.],0) .≈ (0.25,SA[-1,0],SA[0,0.25]))
end

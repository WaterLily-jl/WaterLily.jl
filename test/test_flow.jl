@testset "Flow.jl" begin
    # Test vanLeer
    vanLeer = WaterLily.vanLeer
    @test vanLeer(1,0,1) == 0 && vanLeer(1,2,1) == 2 # larger or smaller than both u,d, reverts to itself
    @test vanLeer(1,2,3) == 2.5 && vanLeer(3,2,1) == 1.5 # if c is between u,d, limiter is quadratic

    # Test central difference scheme
    cds = WaterLily.cds
    @test cds(1,0,1) == 0.5 && cds(1,2,-1) == 0.5 # central difference between downstream and itself

    # Check QUICK scheme on boundary
    ϕuL = WaterLily.ϕuL
    ϕuR = WaterLily.ϕuR
    quick = WaterLily.quick
    ϕ = WaterLily.ϕ

    # inlet with positive flux -> CD
    @test ϕuL(1,CartesianIndex(2),[0.,0.5,2.],1,quick)==ϕ(1,CartesianIndex(2),[0.,0.5,2.0])
    # inlet negative flux -> backward QUICK
    @test ϕuL(1,CartesianIndex(2),[0.,0.5,2.],-1,quick)==-quick(2.0,0.5,0.0)
    # outlet, positive flux -> standard QUICK
    @test ϕuR(1,CartesianIndex(3),[0.,0.5,2.],1,quick)==quick(0.0,0.5,2.0)
    # outlet, negative flux -> backward CD
    @test ϕuR(1,CartesianIndex(3),[0.,0.5,2.],-1,quick)==-ϕ(1,CartesianIndex(3),[0.,0.5,2.0])

    # check that ϕuSelf is the same as ϕu if explicitly provided with the same indices
    ϕu = WaterLily.ϕu
    ϕuP = WaterLily.ϕuP
    λ = WaterLily.quick

    I = CartesianIndex(3); # 1D check, positive flux
    @test ϕu(1,I,[0.,0.5,2.],1,quick)==ϕuP(1,I-2δ(1,I),I,[0.,0.5,2.],1,quick);
    I = CartesianIndex(2); # 1D check, negative flux
    @test ϕu(1,I,[0.,0.5,2.],-1,quick)==ϕuP(1,I-2δ(1,I),I,[0.,0.5,2.],-1,quick);

    # check for periodic flux
    I=CartesianIndex(3);Ip=I-2δ(1,I);
    f = [1.,1.25,1.5,1.75,2.];
    @test ϕuP(1,Ip,I,f,1,quick)==λ(f[Ip],f[I-δ(1,I)],f[I])
    Ip = WaterLily.CIj(1,I,length(f)-2); # make periodic
    @test ϕuP(1,Ip,I,f,1,quick)==λ(f[Ip],f[I-δ(1,I)],f[I])

    # check applying acceleration
    for f ∈ arrays
        p = zeros(4,5) |> f
        apply!(x->x[1]+x[2]+3,p)
        @test inside(p) == CartesianIndices((2:3,2:4))
        @test inside(p,buff=0) == CartesianIndices(p)
        @test L₂(p) == 187
        u = zeros(5,5,2) |> f
        apply!((i,x)->x[i],u)
        @test GPUArrays.@allowscalar [u[i,j,1].-(i-2) for i in 1:3, j in 1:3]==zeros(3,3)
        N = 4; a = zeros(N,N,2) |> f
        WaterLily.accelerate!(a,1,nothing,())
        @test all(a .== 0)
        WaterLily.accelerate!(a,1.,(i,x,t)->i==1 ? t : 2*t,())
        @test all(a[:,:,1] .== 1) && all(a[:,:,2] .== 2)
        WaterLily.accelerate!(a,1.,nothing,(i,x,t) -> i==1 ? -t : -2*t)
        @test all(a[:,:,1] .== 0) && all(a[:,:,2] .== 0)
        WaterLily.accelerate!(a,1.,(i,x,t) -> i==1 ? t : 2*t,(i,x,t) -> i==1 ? -t : -2*t)
        @test all(a[:,:,1] .== 0) && all(a[:,:,2] .== 0)
        # check applying body force (changes in x but not t)
        b = zeros(N,N,2) |> f
        WaterLily.accelerate!(b,0.,(i,x,t)->1,nothing)
        @test all(b .== 1)
        WaterLily.accelerate!(b,1.,(i,x,t)->0,(i,x,t)->t)
        @test all(b .== 2)
        a .= 0; b .= 1 # reset and accelerate using a non-uniform velocity field
        WaterLily.accelerate!(a,0.,nothing,(i,x,t)->t*(x[i]+1.0))
        WaterLily.accelerate!(b,0,(i,x,t)->x[i],nothing)
        @test all(b .== a)
        WaterLily.accelerate!(b,1.,(i,x,t)->x[i]+1.0,nothing)
        WaterLily.accelerate!(a,1.,nothing,(i,x,t)->t*(x[i]+1.0))
        @test all(b .== a)
    end
    # Impulsive flow in a box
    U = (2/3, -1/3)
    N = (2^4, 2^4)
    for mem ∈ arrays
        a = Flow(N, U; mem, T=Float32)
        mom_step!(a, MultiLevelPoisson(a.p,a.μ₀,a.σ))
        @test L₂(a.u[:,:,1].-U[1]) < 2e-5
        @test L₂(a.u[:,:,2].-U[2]) < 1e-5
    end
end

@testset "Flow.jl periodic TGV" begin
    for f ∈ arrays
        sim,TGV = TGVsim(f,T=Float32); ue=copy(sim.flow.u) |> Array
        sim_step!(sim,π/100)
        apply!((i,x)->TGV(i,x,WaterLily.time(sim),2π/sim.L,sim.flow.ν),ue)
        u = sim.flow.u |> Array
        @test WaterLily.L₂(u[:,:,1].-ue[:,:,1]) < 1e-4 &&
              WaterLily.L₂(u[:,:,2].-ue[:,:,2]) < 1e-4
    end
end

@testset "Flow.jl with increasing body force" begin   # acceleratingFlow/gravity! live in helper.jl
    for f ∈ arrays
        N = 8
        sim,jerk = acceleratingFlow(N;use_g=true,mem=f)
        sim_step!(sim,1.0); u = sim.flow.u |> Array
        # Exact uₓ = uₓ₀ + ∫ a dt = uₓ₀ + ∫ jerk*t dt = uₓ₀ + 0.5*jerk*t^2
        uFinal = sim.flow.uBC[1] + 0.5*jerk*WaterLily.time(sim)^2
        @test (
            WaterLily.L₂(u[:,:,1].-uFinal) < 1e-4 &&
            WaterLily.L₂(u[:,:,2].-0) < 1e-4
        )

        # Test with user defined function instead of acceleration
        sim_udf,_ = acceleratingFlow(N;mem=f)
        sim_step!(sim_udf,1.0; udf=gravity!, jerk=jerk); u_udf = sim_udf.flow.u |> Array
        uFinal = sim_udf.flow.uBC[1] + 0.5*jerk*WaterLily.time(sim_udf)^2
        @test (
            WaterLily.L₂(u_udf[:,:,1].-uFinal) < 1e-4 &&
            WaterLily.L₂(u_udf[:,:,2].-0) < 1e-4
        )
    end
end

@testset "Boundary Layer Flow" begin
    for f ∈ arrays
        sim = make_bl_flow(32;mem=f)
        sim_step!(sim,10)
        @test GPUArrays.@allowscalar all(sim.flow.u[1,:,1] .≈ sim.flow.u[end,:,1])
    end
end

@testset "Rotating reference frame" begin
    function rotating_reference(N,x₀::SVector{2,T},ω::T,mem=Array) where T
        function velocity(i,x,t)
            s,c = sincos(ω*t); y = ω*(x-x₀)
            i==1 ? s*y[1]+c*y[2] : -c*y[1]+s*y[2]
        end
        coriolis(i,x,t) = i==1 ? 2ω*velocity(2,x,t) : -2ω*velocity(1,x,t)
        centrifugal(i,x,t) = ω^2*(x-x₀)[i]
        g(i,x,t) = coriolis(i,x,t)+centrifugal(i,x,t)
        udf(a::AbstractFlow,t) = WaterLily.@loop a.f[Ii] += g(last(Ii),loc(Ii,eltype(a.f)),t) over Ii in CartesianIndices(a.f)
        simg = Simulation((N,N),velocity,N; g, U=1, T, mem) # use g
        simg,Simulation((N,N),velocity,N; U=1, T, mem),udf
    end
    L = 4
    simg,sim,udf = rotating_reference(2L,SA_F64[L,L],1/L,Array)
    sim_step!(simg);sim_step!(sim;udf)
    @test L₂(simg.flow.p)==L₂(sim.flow.p)<3e-3 # should be zero
end

@testset "Circle in accelerating flow" begin
    for f ∈ arrays
        make_accel_circle(radius=32,H=16) = Simulation(radius.*(2H,2H),
            (i,x,t)-> i==1 ? t : zero(t), radius; U=1, mem=f,
            body=AutoBody((x,t)->√sum(abs2,x .-H*radius)-radius))
        sim = make_accel_circle(); sim_step!(sim)
        @test isapprox(WaterLily.pressure_force(sim)/(π*sim.L^2),[-1,0],atol=0.04)
        @test GPUArrays.@allowscalar maximum(sim.flow.u)/sim.flow.u[2,2,1] > 1.91 # ≈ 2U
        foreach(i->sim_step!(sim),1:3)
        @test all(sim.pois.n .≤ 2)
        @test !any(isnan.(sim.pois.n))
    end
end

using GPUArrays
using ReadVTK, WriteVTK, JLD2

backend != "KernelAbstractions" && throw(ArgumentError("SIMD backend not allowed to run main tests, use KernelAbstractions backend"))
@info "Test backends: $(join(arrays,", "))"
@testset "util.jl" begin
    I = CartesianIndex(1,2,3,4)
    @test I+δ(3,I) == CartesianIndex(1,2,4,4)
    @test WaterLily.CI(I,5)==CartesianIndex(1,2,3,4,5)
    @test WaterLily.CIj(3,I,5)==CartesianIndex(1,2,5,4)
    @test WaterLily.CIj(2,CartesianIndex(16,16,16,3),14)==CartesianIndex(16,14,16,3)

    @test loc(3,CartesianIndex(3,4,5)) == SVector(3,4,4.5) .- 1.5
    I = CartesianIndex(rand(2:10,3)...)
    @test loc(0,I) == SVector(I.I...) .- 1.5

    ex,sym = :(a[I,i] = Math.add(p.b[I],func(I,q))),[]
    WaterLily.grab!(sym,ex)
    @test ex == :(a[I, i] = Math.add(b[I], func(I, q)))
    # `func` is grabbed as a bare-symbol call head (passed in as a kernel arg) so
    # the @kernel can be lifted to module scope on the KA backend without losing
    # closure-captured callables. `Math.add` (qualified) and operators are skipped.
    @test sym == [:a, :I, :i, :(p.b), :q, :func]
    sym = [:a, :b, :c]
    @test WaterLily.joinsymtype(sym,[:A,:B,:C]) == Expr[:(a::A), :(b::B), :(c::C)]

    for f ∈ arrays
        p = zeros(4,5) |> f
        apply!(x->x[1]+x[2]+3,p) # add 2×1.5 to move edge to origin
        @test inside(p) == CartesianIndices((2:3,2:4))
        @test inside(p,buff=0) == CartesianIndices(p)
        @test L₂(p) == 187

        u = zeros(5,5,2) |> f
        apply!((i,x)->x[i],u)
        @test GPUArrays.@allowscalar [u[i,j,1].-(i-2) for i in 1:3, j in 1:3]==zeros(3,3)

        Ng, D, U = (6, 6), 2, (1.0, 0.5)
        u = rand(Ng..., D) |> f # vector
        σ = rand(Ng...) |> f # scalar
        BC!(u, U)
        @test GPUArrays.@allowscalar all(u[1, :, 1] .== U[1]) && all(u[2, :, 1] .== U[1]) && all(u[end, :, 1] .== U[1]) &&
            all(u[3:end-1, 1, 1] .== u[3:end-1, 2, 1]) && all(u[3:end-1, end, 1] .== u[3:end-1, end-1, 1])
        @test GPUArrays.@allowscalar all(u[:, 1, 2] .== U[2]) && all(u[:, 2, 2] .== U[2]) && all(u[:, end, 2] .== U[2]) &&
            all(u[1, 3:end-1, 2] .== u[2, 3:end-1, 2]) && all(u[end, 3:end-1, 2] .== u[end-1, 3:end-1, 2])

        GPUArrays.@allowscalar u[end,:,1] .= 3
        BC!(u,U,true) # save exit values
        @test GPUArrays.@allowscalar all(u[end, :, 1] .== 3)

        WaterLily.exitBC!(u,u,0) # conservative exit check
        @test GPUArrays.@allowscalar all(u[end,2:end-1, 1] .== U[1])

        # test BC with function
        Ubc(i,x,t) = i==1 ? 1.0 : 0.5
        v = rand(Ng..., D) |> f # vector
        BC!(v,Ubc,false); BC!(u,U,false) # make sure we apply the same
        @test GPUArrays.@allowscalar all(v[1, :, 1] .== u[1, :, 1]) && all(v[2, :, 1] .== u[2, :, 1]) && all(v[end, :, 1] .== u[end, :, 1])
        @test GPUArrays.@allowscalar all(v[:, 1, 2] .== u[:, 1, 2]) && all(v[:, 2, 2] .== u[:, 2, 2]) && all(v[:, end, 2] .== u[:, end, 2])
        # test exit bc
        GPUArrays.@allowscalar v[end,:,1] .= 3
        BC!(v,Ubc,true) # save exit values
        @test GPUArrays.@allowscalar all(v[end, :, 1] .== 3)

        BC!(u,U,true,(2,)) # periodic in y and save exit values
        @test GPUArrays.@allowscalar all(u[:, 1:2, 1] .== u[:, end-1:end, 1]) && all(u[:, 1:2, 1] .== u[:,end-1:end,1])
        WaterLily.perBC!(σ,(1,2)) # periodic in two directions
        @test GPUArrays.@allowscalar all(σ[1, 2:end-1] .== σ[end-1, 2:end-1]) && all(σ[2:end-1, 1] .== σ[2:end-1, end-1])

        u = rand(Ng..., D) |> f # vector
        BC!(u,U,true,(1,)) #saveexit has no effect here as x-periodic
        @test GPUArrays.@allowscalar all(u[1:2, :, 1] .== u[end-1:end, :, 1]) && all(u[1:2, :, 2] .== u[end-1:end, :, 2]) &&
                           all(u[:, 1, 2] .== U[2]) && all(u[:, 2, 2] .== U[2]) && all(u[:, end, 2] .== U[2])
        # test non-uniform BCs
        Ubc_1(i,x,t) = i==1 ? x[2] : x[1]
        v .= 0; BC!(v,Ubc_1)
        # the tangential BC change the value of the ghost cells on the other axis, so we cannot check it
        @test GPUArrays.@allowscalar all(v[1,2:end-1,1] .≈ v[end,2:end-1,1])
        @test GPUArrays.@allowscalar all(v[2:end-1,1,2] .≈ v[2:end-1,end,2])
        # more complex
        Ng, D = (8, 8, 8), 3
        u = zeros(Ng..., D) |> f # vector
        Ubc_2(i,x,t) = i==1 ? cos(2π*x[1]/8) : i==2 ? sin(2π*x[2]/8) : tan(π*x[3]/16)
        BC!(u,Ubc_2)
        @test GPUArrays.@allowscalar all(u[1,:,:,1] .≈ cos(-1π/4))  && all(u[2,:,:,1] .≈ cos(0)) && all(u[end,:,:,1] .≈ cos(6π/4))
        @test GPUArrays.@allowscalar all(u[:,1,:,2] .≈ sin(-1π/4))  && all(u[:,2,:,2] .≈ sin(0)) && all(u[:,end,:,2] .≈ sin(6π/4))
        @test GPUArrays.@allowscalar all(u[:,:,1,3] .≈ tan(-1π/16)) && all(u[:,:,2,3] .≈ tan(0)) && all(u[:,:,end,3].-tan(6π/16).<1e-6)

       # test interpolation, test on two different array type
       a = zeros(Float32,8,8,2) |> f; b = zeros(Float64,8,8) |> f
       apply!((i,x)->x[i],a); apply!(x->x[1],b) # offset for start of grid
       @test GPUArrays.@allowscalar all(WaterLily.interp(SVector(2.5f0,1.f0),a) .≈ [2.5f0,1.0f0])
       @test GPUArrays.@allowscalar all(WaterLily.interp(SVector(3.5f0,3.f0),a) .≈ [3.5f0,3.0f0])
       @test GPUArrays.@allowscalar eltype(WaterLily.interp(SVector(2.5f0,1.f0),a))==Float32
       @test_throws MethodError GPUArrays.@allowscalar WaterLily.interp(SVector(2.50,1.0),a)
       @test GPUArrays.@allowscalar WaterLily.interp(SVector(2.5,1),b) ≈ 2.5
       @test GPUArrays.@allowscalar WaterLily.interp(SVector(3.5,3),b) ≈ 3.5
       @test GPUArrays.@allowscalar eltype(WaterLily.interp(SVector(3.5,3),b))==Float64
       @test_throws MethodError GPUArrays.@allowscalar WaterLily.interp(SVector(2.5f0,1.f0),b)

        # test on perdot
        σ1 = rand(Ng...) |> f # scalar
        σ2 = rand(Ng...) |> f # another scalar 
        # use ≈ instead of == as summation in different order might result in slight difference in floating point expressions
        @test GPUArrays.@allowscalar WaterLily.perdot(σ1,σ2,())    ≈ sum(σ1[I]*σ2[I] for I∈CartesianIndices(σ1))
        @test GPUArrays.@allowscalar WaterLily.perdot(σ1,σ2,(1,))  ≈ sum(σ1[I]*σ2[I] for I∈inside(σ1))
        @test GPUArrays.@allowscalar WaterLily.perdot(σ1,σ2,(1,2)) ≈ sum(σ1[I]*σ2[I] for I∈inside(σ1))
    end
end

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
        @test err < 1e-6
        @test pois.n[] < 310
        err,pois = Poisson_setup(Poisson,(2^4+2,2^4+2,2^4+2);f)
        @test err < 1e-6
        @test pois.n[] < 35
    end
end

@testset "MultiLevelPoisson.jl" begin
    I = CartesianIndex(4,3,2)
    @test all(WaterLily.down(J)==I for J ∈ WaterLily.up(I))
    @test_throws AssertionError("MultiLevelPoisson requires size=a2ⁿ, where n>2") Poisson_setup(MultiLevelPoisson,(15+2,3^4+2))

    err,pois = Poisson_setup(MultiLevelPoisson,(10,10))
    @test pois.levels[3].D == Float32[0 0 0 0; 0 -2 -2 0; 0 -2 -2 0; 0 0 0 0]
    @test err < 1e-5

    pois.levels[1].L[5:6,:,1].=0
    WaterLily.update!(pois)
    @test pois.levels[3].D == Float32[0 0 0 0; 0 -1 -1 0; 0 -1 -1 0; 0 0 0 0]

    for f ∈ arrays
        err,pois = Poisson_setup(MultiLevelPoisson,(2^6+2,2^6+2);f)
        @test err < 1e-6
        @test pois.n[] ≤ 3
        err,pois = Poisson_setup(MultiLevelPoisson,(2^4+2,2^4+2,2^4+2);f)
        @test err < 1e-6
        @test pois.n[] ≤ 3
    end
end

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
    for f ∈ arrays
        a = Flow(N, U; f, T=Float32)
        mom_step!(a, MultiLevelPoisson(a.p,a.μ₀,a.σ))
        @test L₂(a.u[:,:,1].-U[1]) < 2e-5
        @test L₂(a.u[:,:,2].-U[2]) < 1e-5
    end
end

@testset "Body.jl" begin
    @test WaterLily.μ₀(3,6)==WaterLily.μ₀(0.5,1)
    @test WaterLily.μ₀(0,1)==0.5
    @test WaterLily.μ₁(0,2)==2*(1/4-1/π^2)

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

function TGVsim(mem;perdir=(1,2),Re=1e8,T=typeof(Re))
    # Define vortex size, velocity, viscosity
    L = 64; κ = T(2π/L); ν = T(1/(κ*Re));
    # TGV vortex in 2D
    function TGV(i,xy,t,κ,ν)
        x,y = @. (xy)*κ  # scaled coordinates
        i==1 && return -sin(x)*cos(y)*exp(-2*κ^2*ν*t) # u_x
        return          cos(x)*sin(y)*exp(-2*κ^2*ν*t) # u_y
    end
    # Initialize simulation
    return Simulation((L,L),(i,x,t)->TGV(i,x,t,κ,ν),L;U=1,ν,T,mem,perdir),TGV
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
@testset "ForwardDiff" begin
    function TGV_ke(Re)
        sim,_ = TGVsim(Array;Re)
        sim_step!(sim,π/100)
        sum(I->WaterLily.ke(I,sim.flow.u),inside(sim.flow.p))
    end
    using ForwardDiff:derivative
    @test derivative(TGV_ke,1e2) ≈ (TGV_ke(1e2+1)-TGV_ke(1e2-1))/2 rtol=1e-1

    # Spinning cylinder lift generation
    rot(θ) = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]  # rotation matrix
    function spinning(ξ;D=16,Re=500)
        C,R,U = SA[D,D],D÷2,1
        body = AutoBody((x,t)->√(x'*x)-R,          # circle sdf
                        (x,t)->rot(ξ*U*t/R)*(x-C)) # center & spin!
        Simulation((2D,2D),(U,0),D;ν=U*D/Re,body,T=typeof(ξ))
    end
    function lift(ξ,t_end=1)
        sim = spinning(ξ)
        sim_step!(sim,t_end;remeasure=false)
        WaterLily.total_force(sim)[2]/(ξ^2*sim.U^2*sim.L)
    end
    h = 1e-6
    @test derivative(lift,2.0) ≈ (lift(2+h)-lift(2-h))/2h rtol=√h
end

function acceleratingFlow(N;use_g=false,T=Float64,perdir=(1,),jerk=4,mem=Array)
    # periodic in x, Neumann in y
    # assuming gravitational scale is 1 and Fr is 1, U scale is Fr*√gL
    UScale = √N  # this is also initial U
    # constant jerk in x, zero acceleration in y
    g(i,x,t) = i==1 ? t*jerk : 0.
    !use_g && (g = nothing)
    return WaterLily.Simulation(
        (N,N), (UScale,0.), N; ν=0.001,g,Δt=0.001,perdir,T,mem
    ),jerk
end
gravity!(flow::Flow,t; jerk=4) = for i ∈ 1:last(size(flow.f))
    WaterLily.@loop flow.f[I,i] += i==1 ? t*jerk : 0 over I ∈ CartesianIndices(Base.front(size(flow.f)))
end
@testset "Flow.jl with increasing body force" begin
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

make_bl_flow(L=32;T=Float32,mem=Array) = Simulation((L,L),
    (i,x,t)-> i==1 ? convert(Float32,4.0*(((x[2]+0.5)/2L)-((x[2]+0.5)/2L)^2)) : 0.f0,
    L;ν=0.001,U=1,mem,T,exitBC=false
) # fails with exitBC=true, but the profile is maintained
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
        udf(a::Flow,t) = WaterLily.@loop a.f[Ii] += g(last(Ii),loc(Ii,eltype(a.f)),t) over Ii in CartesianIndices(a.f)
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

import WaterLily: ×
@testset "Metrics.jl" begin
    J = CartesianIndex(2,3,4); x = loc(0,J,Float64); px = prod(x)
    for f ∈ arrays
        u = zeros(3,4,5,3) |> f; apply!((i,x)->x[i]+prod(x),u)
        p = zeros(3,4,5) |> f
        @inside p[I] = WaterLily.ke(I,u)
        @test GPUArrays.@allowscalar p[J]==0.5*sum(abs2,x .+ px)
        @inside p[I] = WaterLily.ke(I,u,x)
        @test GPUArrays.@allowscalar p[J]==1.5*px^2
        @inside p[I] = WaterLily.λ₂(I,u)
        @test GPUArrays.@allowscalar p[J]≈1
        ω = (1 ./ x)×repeat([px],3)
        @inside p[I] = WaterLily.curl(2,I,u)
        @test GPUArrays.@allowscalar p[J]==ω[2]
        f==Array && @test WaterLily.ω(J,u)≈ω
        @inside p[I] = WaterLily.ω_mag(I,u)
        @test GPUArrays.@allowscalar p[J]==sqrt(sum(abs2,ω))
        @inside p[I] = WaterLily.ω_θ(I,(0,0,1),x .+ (0,1,2),u)
        @test GPUArrays.@allowscalar p[J]≈ω[1]
        apply!((x)->1,p)
        @test WaterLily.L₂(p)≈prod(size(p).-2)
        # test force routines
        N = 32
        p = zeros(N,N) |> f; df₂ = zeros(N,N,2) |> f; df₃ = zeros(N,N,N,3) |> f
        @inside p[I] = loc(0, I, eltype(p))[2]
        body = AutoBody((x,t)->√sum(abs2,x.-(N/2))-N÷4,(x,t)->x)
        force = WaterLily.pressure_force(p,df₂,body)
        @test sum(abs,force/(π*(N/4)^2) - [0,1]) < 2e-3
        # stress tensor
        u₂ = zeros(N,N,2) |> f
        u₃ = zeros(N,N,N,3) |> f
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2),u₂) .≈ 0)
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2,N÷2),u₃) .≈ 0)
        apply!((i,x)->x[i],u₂) # uniform gradient
        apply!((i,x)->x[i],u₃)
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2),u₂) .≈ SA[2 0; 0 2])
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2,N÷2),u₃) .≈ SA[2 0 0; 0 2 0; 0 0 2])
        apply!((i,x)->x[i%2+1],u₂) # shear
        apply!((i,x)->x[i%3+1],u₃)
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2),u₂) .≈ SA[0 2; 2 0])
        @test GPUArrays.@allowscalar all(2WaterLily.S(CartesianIndex(N÷2,N÷2,N÷2),u₃) .≈ SA[0 1 1; 1 0 1; 1 1 0])
        # viscous force
        u₂ .= 0; u₃ .= 0
        @test all(WaterLily.viscous_force(u₂,1.0,df₂,body) .≈ 0)
        @test all(WaterLily.viscous_force(u₃,1.0,df₃,body) .≈ 0)
        # pressure moment
        p₂ = zeros(N,N) |> f; apply!(x->x[2],p₂)
        p₃ = zeros(N,N,N) |> f; apply!(x->x[2],p₃)
        @test WaterLily.pressure_moment(SVector{2,Float64}(N/2,N/2),p₂,df₂,body,0)[1] ≈ 0 # no moment in hydrostatic pressure
        @test all(WaterLily.pressure_moment(SVector{3,Float64}(N/2,N/2,N/2),p₃,df₃,body,0) .≈ SA[0 0 0]) # with a 3D field, 3D moments
        # temporal averages
        T = Float32
        sim = make_bl_flow(; T, mem=f)
        meanflow = MeanFlow(sim.flow; uu_stats=true)
        for t in range(0,10;step=0.1)
            sim_step!(sim, t)
            update!(meanflow, sim.flow)
        end
        @test all(isapprox.(Array(sim.flow.u), Array(meanflow.U); atol=√eps(T))) # can't broadcast isapprox for GPUArrays...
        @test all(isapprox.(Array(sim.flow.p), Array(meanflow.P); atol=√eps(T)))
        for i in 1:ndims(sim.flow.p), j in 1:ndims(sim.flow.p)
            @test all(isapprox.(Array(sim.flow.u)[:,:,i] .* Array(sim.flow.u)[:,:,j], Array(meanflow.UU)[:,:,i,j]; atol=√eps(T)))
        end
        τ = uu(meanflow)
        for i in 1:ndims(sim.flow.p), j in 1:ndims(sim.flow.p)
            @test all(isapprox.(
                Array(meanflow.UU)[:,:,i,j] .- Array(meanflow.U)[:,:,i].*Array(meanflow.U)[:,:,j],
                Array(τ)[:,:,i,j]; atol=√eps(T))
            )
        end
        @test WaterLily.time(sim.flow) == WaterLily.time(meanflow)
        WaterLily.reset!(meanflow)
        @test all(meanflow.U .== zero(T))
        @test all(meanflow.P .== zero(T))
        @test all(meanflow.UU .== zero(T))
        @test meanflow.t == T[0]

        meanflow2 = MeanFlow(size(sim.flow.p).-2; uu_stats=true)
        @test all(meanflow2.P .== zero(T))
        @test size(meanflow2.P) == size(meanflow.P)
    end
end

@testset "WaterLily.jl" begin
    radius = 8; ν=radius/250; T=Float32; nm = radius.*(4,4)
    circle(x,t) = √sum(abs2,x .- 2radius) - radius
    move(x,t) = x-SA[t,0]
    accel(x,t) = x-SA[2t^2,0]
    plate(x,t) = √sum(abs2,x - SA[clamp(x[1],-radius+2,radius-2),0])-2
    function rotate(x,t)
        s,c = sincos(t/radius+1); R = SA[c s ; -s c]
        R * (x .- 2radius)
    end
    function bend(xy,t) # into ≈ circular arc
        x,y = xy .- 2radius; κ = 2t/radius^2+0.2f0/radius
        return SA[x+x^3*κ^2/6,y-x^2*κ/2]
    end
    # Test sim_time, and sim_step! stopping time
    sim = Simulation(nm,(1,0),radius; body=AutoBody(circle), ν, T)
    @test sim_time(sim) == 0
    sim_step!(sim,0.1,remeasure=false)
    @test sim_time(sim) ≥ 0.1 > sum(sim.flow.Δt[1:end-2])*sim.U/sim.L
    for mem ∈ arrays, exitBC ∈ (true,false)
        # Test that remeasure works perfectly when V = U = 1
        sim = Simulation(nm,(1,0),radius; body=AutoBody(circle,move), ν, T, mem, exitBC)
        sim_step!(sim)
        @test all(sim.flow.u[:,radius,1].≈1)
        # @test all(sim.pois.n .== 0)
        # Test accelerating from U=0 to U=1
        sim = Simulation(nm,(0,0),radius; U=1, body=AutoBody(circle,accel), ν, T, mem, exitBC)
        sim_step!(sim)
        @test length(sim.pois.n)==2 && all(sim.pois.n .<5)
        @test maximum(sim.flow.u) > maximum(sim.flow.V) > 0
        # Test that non-uniform V doesn't break
        sim = Simulation(nm,(0,0),radius; U=1, body=AutoBody(plate,rotate), ν, T, mem, exitBC)
        sim_step!(sim)
        @test length(sim.pois.n)==2 && all(sim.pois.n .<5)
        @test 1 > sim.flow.Δt[end] > 0.5
        # Test that divergent V doesn't break
        sim = Simulation(nm,(0,0),radius; U=1, body=AutoBody(plate,bend), ν, T, mem, exitBC)
        sim_step!(sim)
        @test length(sim.pois.n)==2 && all(sim.pois.n .<5)
        @test 1.2 > sim.flow.Δt[end] > 0.8
    end
end

function sphere_sim(radius = 8; D=2, mem=Array, exitBC=false)
    body = AutoBody((x,t)-> √sum(abs2,x .- (2radius+1.5)) - radius)
    D==2 && Simulation(radius.*(6,4),(1,0),radius; body, ν=radius/250, T=Float32, mem, exitBC)
    Simulation(radius.*(6,4,1),(1,0,0),radius; body, ν=radius/250, T=Float32, mem, exitBC)
end
@testset "VTKExt.jl" begin
    for D ∈ [2,3], mem ∈ arrays
        # make a simulation
        sim = sphere_sim(;D,mem);
        # make a vtk writer
        wr = vtkWriter("test_vtk_reader_$D";dir="TEST_DIR")
        sim_step!(sim,1); save!(wr, sim); close(wr)

        # re start the sim from a paraview file
        restart = sphere_sim(;D,mem);
        load!(restart; fname="test_vtk_reader_$D.pvd")

        # check that the restart is the same as the original
        @test all(sim.flow.p .== restart.flow.p)
        @test all(sim.flow.u .== restart.flow.u)
        @test all(sim.flow.μ₀ .== restart.flow.μ₀)
        @test sim.flow.Δt[end] == restart.flow.Δt[end]
        @test abs(sim_time(sim)-sim_time(restart))<1e-3

        # clean-up
        @test_nowarn rm("TEST_DIR",recursive=true)
        @test_nowarn rm("test_vtk_reader_$D.pvd")
    end
end

@testset "WaterLilyJLD2Ext.jl" begin
    test_dir = "TEST_DIR"; mkpath(test_dir)
    for D ∈ [2,3], mem ∈ arrays
        sim1 = sphere_sim(;D,mem)
        sim_step!(sim1, 1)
        save!("sim1_sphere.jld2", sim1; dir=test_dir)

        sim2 = sphere_sim(;D,mem)
        load!(sim2; fname="sim1_sphere.jld2", dir=test_dir)

        @test all(sim1.flow.p .== sim2.flow.p)
        @test all(sim1.flow.u .== sim2.flow.u)
        @test all(sim1.flow.Δt .== sim2.flow.Δt)

        # temporal averages
        sim = make_bl_flow(; T=Float32, mem)
        meanflow1 = MeanFlow(sim.flow; uu_stats=true)
        for t in range(0,10;step=0.1)
            sim_step!(sim, t)
            update!(meanflow1, sim.flow)
        end
        save!("meanflow.jld2", meanflow1; dir=test_dir)
        meanflow2 = MeanFlow(sim.flow; uu_stats=true)
        WaterLily.reset!(meanflow2)
        load!(meanflow2; fname="meanflow.jld2", dir=test_dir)
        @test all(meanflow1.U .== meanflow2.U)
        @test all(meanflow1.P .== meanflow2.P)
        @test all(meanflow1.UU .== meanflow2.UU)
        @test all(meanflow1.t .== meanflow2.t)
    end
    @test_nowarn rm(test_dir, recursive=true)
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
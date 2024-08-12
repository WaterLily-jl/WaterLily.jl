using GPUArrays
using ReadVTK, WriteVTK
using FileIO,JLD2

@info "Test backends: $(join(arrays,", "))"
@testset "util.jl" begin
    I = CartesianIndex(1,2,3,4)
    @test I+δ(3,I) == CartesianIndex(1,2,4,4)
    @test WaterLily.CI(I,5)==CartesianIndex(1,2,3,4,5)
    @test WaterLily.CIj(3,I,5)==CartesianIndex(1,2,5,4)
    @test WaterLily.CIj(2,CartesianIndex(16,16,16,3),14)==CartesianIndex(16,14,16,3)

    @test loc(3,CartesianIndex(3,4,5)) == SVector(3,4,4.5) .- 2.5
    I = CartesianIndex(rand(2:10,3)...)
    @test loc(0,I) == SVector(I.I...) .- 2.5

    ex,sym = :(a[I,i] = Math.add(p.b[I],func(I,q))),[]
    WaterLily.grab!(sym,ex)
    @test ex == :(a[I, i] = Math.add(b[I], func(I, q)))
    @test sym == [:a, :I, :i, :(p.b), :q]

    for f ∈ arrays
        p = zeros(6,7) |> f
        apply!(x->x[1]+x[2]+3,p) # add 2×1.5 to move edge to origin
        @test inside(p) == CartesianIndices((3:4,3:5))
        @test inside(p,buff=0) == CartesianIndices(p)
        @test L₂(p) == 187

        u = zeros(5,5,2) |> f
        apply!((i,x)->x[i],u)
        @test GPUArrays.@allowscalar [u[i,j,1].-(i-3) for i in 1:3, j in 1:3]==zeros(3,3)

        Ng, D, U = (6, 6), 2, (1.0, 0.5)
        u = rand(Ng..., D) |> f # vector
        σ = rand(Ng...) |> f # scalar
        BC!(u, U)
        @test GPUArrays.@allowscalar all(u[1, :, 1] .== U[1]) && all(u[2, :, 1] .== U[1]) && all(u[end, :, 1] .== U[1]) &&
            all(u[3:end-1, 1, 1] .== u[3:end-1, 2, 1]) && all(u[3:end-1, end, 1] .== u[3:end-1, end-1, 1])
        @test GPUArrays.@allowscalar all(u[:, 1, 2] .== U[2]) && all(u[:, 2, 2] .== U[2]) && all(u[:, end, 2] .== U[2]) &&
            all(u[1, 3:end-1, 2] .== u[2, 3:end-1, 2]) && all(u[end, 3:end-1, 2] .== u[end-1, 3:end-1, 2])

        GPUArrays.@allowscalar u[end-1,:,1] .= 3
        BC!(u,U,true) # save exit values
        @test GPUArrays.@allowscalar all(u[end-1, :, 1] .== 3)

        WaterLily.exitBC!(u,u,U,0) # conservative exit check
        @test GPUArrays.@allowscalar all(u[end-1,3:end-2, 1] .== U[1])

        BC!(u,U,true,(2,)) # periodic in y and save exit values
        @test GPUArrays.@allowscalar all(u[:, 1:2, 1] .== u[:, end-1:end, 1]) && all(u[:, 1:2, 1] .== u[:,end-1:end,1])
        WaterLily.perBC!(σ,(1,2)) # periodic in two directions
        @test GPUArrays.@allowscalar all(σ[1, 2:end-1] .== σ[end-1, 2:end-1]) && all(σ[2:end-1, 1] .== σ[2:end-1, end-1])

        u = rand(Ng..., D) |> f # vector
        BC!(u,U,true,(1,)) #saveexit has no effect here as x-periodic
        @test GPUArrays.@allowscalar all(u[1:2, :, 1] .== u[end-1:end, :, 1]) && all(u[1:2, :, 2] .== u[end-1:end, :, 2]) &&
                           all(u[:, 1, 2] .== U[2]) && all(u[:, 2, 2] .== U[2]) && all(u[:, end, 2] .== U[2])

        # test interpolation
        a = zeros(5,5,2) |> f; b = zeros(5,5) |> f
        apply!((i,x)->x[i]+1.5,a); apply!(x->x[1]+1.5,b) # offset for start of grid
        @test GPUArrays.@allowscalar all(WaterLily.interp(SVector(2.5,1),a) .≈ [1.5,0.])
        @test GPUArrays.@allowscalar all(WaterLily.interp(SVector(3.5,3),a) .≈ [2.5,2.])
        @test GPUArrays.@allowscalar WaterLily.interp(SVector(2.5,1),b) ≈ 1.5
        @test GPUArrays.@allowscalar WaterLily.interp(SVector(3.5,3),b) ≈ 2.5
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
        err,pois = Poisson_setup(Poisson,(7,7);f)
        @test GPUArrays.@allowscalar parent(pois.D)==f(Float32[0 0 0 0 0 0 0;0 0 0 0 0 0 0;0 0 -2 -3 -2 0 0;0 0 -3 -4 -3 0 0;0 0 -2 -3 -2 0 0; 0 0 0 0 0 0 0;0 0 0 0 0 0 0])
        @test GPUArrays.@allowscalar parent(pois.iD)≈f(Float32[0 0 0 0 0 0 0;0 0 0 0 0 0 0;0 0 -1/2 -1/3 -1/2 0 0;0 0 -1/3 -1/4 -1/3 0 0;0 0 -1/2 -1/3 -1/2 0 0;0 0 0 0 0 0 0;0 0 0 0 0 0 0])
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

    err,pois = Poisson_setup(MultiLevelPoisson,(12,12))
    @test pois.levels[3].D == Float32[0 0 0 0 0 0;0 0 0 0 0 0;0 0 -2 -2 0 0;
                                      0 0 -2 -2 0 0;0 0 0 0 0 0;0 0 0 0 0 0]
    @test err < 1e-5

    pois.levels[1].L[5:6,:,1].=0
    WaterLily.update!(pois)
    @test pois.levels[3].D == Float32[0 0 0 0 0 0;0 0 0 0 0 0;0 0 -2 -2 0 0;
                                      0 0 -2 -2 0 0;0 0 0 0 0 0;0 0 0 0 0 0]

    for f ∈ arrays
        err,pois = Poisson_setup(MultiLevelPoisson,(2^6+4,2^6+4);f)
        @test err < 1e-6
        @test pois.n[] ≤ 3
        err,pois = Poisson_setup(MultiLevelPoisson,(2^4+4,2^4+4,2^4+4);f)
        @test err < 1e-6
        @test pois.n[] ≤ 3
    end
end

@testset "Flow.jl" begin
    # test than vanLeer behaves correctly
    vanLeer = WaterLily.vanLeer
    @test vanLeer(1,0,1) == 0 && vanLeer(1,2,1) == 2 # larger or smaller than both u,d revetrs to itlsef
    @test vanLeer(1,2,3) == 2.5 && vanLeer(3,2,1) == 1.5 # if c is between u,d, limiter is quadratic

    @test all(WaterLily.BCTuple((1,2,3),[0],3).==WaterLily.BCTuple((i,t)->i,0,3))
    @test all(WaterLily.BCTuple((i,t)->t,[1.234],3).==ntuple(i->1.234,3))

    # check applying acceleration
    for f ∈ arrays
        N = 4; a = zeros(N,N,2) |> f
        WaterLily.accelerate!(a,[1],nothing,())
        @test all(a .== 0)
        WaterLily.accelerate!(a,[1],(i,t) -> i==1 ? t : 2*t,())
        @test all(a[:,:,1] .== 1) && all(a[:,:,2] .== 2)
        WaterLily.accelerate!(a,[1],nothing,(i,t) -> i==1 ? -t : -2*t)
        @test all(a[:,:,1] .== 0) && all(a[:,:,2] .== 0)
        WaterLily.accelerate!(a,[1],(i,t) -> i==1 ? t : 2*t,(i,t) -> i==1 ? -t : -2*t)
        @test all(a[:,:,1] .== 0) && all(a[:,:,2] .== 0)
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
end

@testset "AutoBody.jl" begin
    norm2(x) = √sum(abs2,x)
    # test AutoDiff in 2D and 3D
    body1 = AutoBody((x,t)->norm2(x)-2-t)
    @test all(measure(body1,[√2.,√2.],0.).≈(0,[√.5,√.5],[0.,0.]))
    @test all(measure(body1,[2.,0.,0.],1.).≈(-1.,[1.,0.,0.],[0.,0.,0.]))
    body2 = AutoBody((x,t)->norm2(x)-2,(x,t)->x.+t^2)
    @test all(measure(body2,[√2.,√2.],0.).≈(0,[√.5,√.5],[0.,0.]))
    @test all(measure(body2,[1.,-1.,-1.],1.).≈(0.,[1.,0.,0.],[-2.,-2.,-2.]))

    # test booleans
    @test all(measure(body1+body2,[-√2.,-√2.],1.).≈(-√2.,[-√.5,-√.5],[-2.,-2.]))
    @test all(measure(body1∪body2,[-√2.,-√2.],1.).≈(-√2.,[-√.5,-√.5],[-2.,-2.]))
    @test all(measure(body1-body2,[-√2.,-√2.],1.).≈(√2.,[√.5,√.5],[-2.,-2.]))

    # tests for Bodies
    @test all(measure(Bodies([body1,body2]),[-√2.,-√2.],1.).≈measure(body1+body2,[-√2.,-√2.],1.))
    @test all(measure(Bodies([body1,body2],-),[-√2.,-√2.],1.).≈measure(body1-body2,[-√2.,-√2.],1.))

    radius = [1.0, 0.75, 0.5, 0.25]
    circles = [(x,t) -> √sum(abs2,x)-r for r ∈ radius]
    body = AutoBody(circles[1])-AutoBody(circles[2])+AutoBody(circles[3])-AutoBody(circles[4])
    bodies = Bodies(AutoBody[AutoBody(c) for c ∈ circles], [-,+,-])
    xy = rand(2)
    @test all(measure(body, xy, 1.).≈measure(bodies, xy, 1.))

    # test curvature, 2D and 3D
    # A = ForwardDiff.Hessian(y->body1.sdf(y,0.0),[0.,0.])
    @test all(WaterLily.curvature([1. 0.; 0. 1.]).≈(1.,0.))
    @test all(WaterLily.curvature([2. 1. 0.; 1. 2. 1.; 0. 1. 2.]).≈(3.,10.))

    # check that sdf functions are the same
    for f ∈ arrays
        p = zeros(6,7) |> f; measure_sdf!(p,body1)
        I = CartesianIndex(4,5)
        @test GPUArrays.@allowscalar p[I]≈body1.sdf(loc(0,I,eltype(p)),0.0)
    end

    # check fast version
    @test all(measure(body1,[3.,4.],0.,fastd²=9) .≈ measure(body1,[3.,4.],0.))
    @test all(measure(body1,[3.,4.],0.,fastd²=8) .≈ (sdf(body1,[3.,4.],0.,fastd²=9),zeros(2),zeros(2)))
end

function TGVsim(mem;perdir=(1,2),Re=1e8,T=typeof(Re))
    # Define vortex size, velocity, viscosity
    L = 64; κ=2π/L; ν = 1/(κ*Re);
    # TGV vortex in 2D
    function TGV(i,xy,t,κ,ν)
        x,y = @. (xy)*κ  # scaled coordinates
        i==1 && return -sin(x)*cos(y)*exp(-2*κ^2*ν*t) # u_x
        return          cos(x)*sin(y)*exp(-2*κ^2*ν*t) # u_y
    end
    # Initialize simulation
    return Simulation((L,L),(0,0),L;U=1,uλ=(i,x)->TGV(i,x,0.0,κ,ν),ν,T,mem,perdir),TGV
end
@testset "Flow.jl periodic TGV" begin
    for f ∈ arrays
        sim,TGV = TGVsim(f,T=Float32); ue=copy(sim.flow.u) |> f
        sim_step!(sim,π/100)
        apply!((i,x)->TGV(i,x,WaterLily.time(sim),2π/sim.L,sim.flow.ν),ue)
        u = sim.flow.u |> f
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
    # @test derivative(TGV_ke,1e3) ≈ (TGV_ke(1e3+1)-TGV_ke(1e3-1))/2 rtol=1e-6

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

function acceleratingFlow(N;T=Float64,perdir=(1,),jerk=4,mem=Array)
    # periodic in x, Neumann in y
    # assuming gravitational scale is 1 and Fr is 1, U scale is Fr*√gL
    UScale = √N  # this is also initial U
    # constant jerk in x, zero acceleration in y
    g(i,t) = i==1 ? t*jerk : 0
    return WaterLily.Simulation(
        (N,N), (UScale,0.), N; ν=0.001,g,Δt=0.001,perdir,T,mem
    ),jerk
end
@testset "Flow.jl with increasing body force" begin
    for f ∈ arrays
        N = 8
        sim,jerk = acceleratingFlow(N;mem=f)
        sim_step!(sim,1.0); u = sim.flow.u |> Array
        # Exact uₓ = uₓ₀ + ∫ a dt = uₓ₀ + ∫ jerk*t dt = uₓ₀ + 0.5*jerk*t^2
        uFinal = sim.flow.U[1] + 0.5*jerk*WaterLily.time(sim)^2
        @test (
            WaterLily.L₂(u[:,:,1].-uFinal) < 1e-4 &&
            WaterLily.L₂(u[:,:,2].-0) < 1e-4
        )
    end
end
import WaterLily: ×
@testset "Metrics.jl" begin
    J = CartesianIndex(3,4,3); x = loc(0,J,Float64); px = prod(x)
    for f ∈ arrays
        u = zeros(5,6,7,3) |> f; apply!((i,x)->x[i]+prod(x),u)
        p = zeros(5,6,7) |> f
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
        @test WaterLily.L₂(p)≈prod(size(p).-4)
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
        @test GPUArrays.@allowscalar all(WaterLily.∇²u(CartesianIndex(N÷2,N÷2),u₂) .≈ 0)
        @test GPUArrays.@allowscalar all(WaterLily.∇²u(CartesianIndex(N÷2,N÷2,N÷2),u₃) .≈ 0)
        apply!((i,x)->x[i],u₂) # uniform gradient
        apply!((i,x)->x[i],u₃)
        @test GPUArrays.@allowscalar all(WaterLily.∇²u(CartesianIndex(N÷2,N÷2),u₂) .≈ SA[2 0; 0 2])
        @test GPUArrays.@allowscalar all(WaterLily.∇²u(CartesianIndex(N÷2,N÷2,N÷2),u₃) .≈ SA[2 0 0; 0 2 0; 0 0 2])
        apply!((i,x)->x[i%2+1],u₂) # shear
        apply!((i,x)->x[i%3+1],u₃)
        @test GPUArrays.@allowscalar all(WaterLily.∇²u(CartesianIndex(N÷2,N÷2),u₂) .≈ SA[0 2; 2 0])
        @test GPUArrays.@allowscalar all(WaterLily.∇²u(CartesianIndex(N÷2,N÷2,N÷2),u₃) .≈ SA[0 1 1; 1 0 1; 1 1 0])
        # viscous force
        u₂ .= 0; u₃ .= 0
        @test all(WaterLily.viscous_force(u₂,1.0,df₂,body) .≈ 0)
        @test all(WaterLily.viscous_force(u₃,1.0,df₃,body) .≈ 0)
        # pressure moment
        p₂ = zeros(N,N) |> f; apply!(x->x[2],p₂)
        p₃ = zeros(N,N,N) |> f; apply!(x->x[2],p₃)
        @test WaterLily.pressure_moment(SVector{2,Float64}(N/2,N/2),p₂,df₂,body,0)[1] ≈ 0 # no moment in hydrostatic pressure
        @test all(WaterLily.pressure_moment(SVector{3,Float64}(N/2,N/2,N/2),p₃,df₃,body,0) .≈ SA[0 0 0]) # with a 3D field, 3D moments
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
        @test sim.pois.n == [2,2]
        @test maximum(sim.flow.u) > maximum(sim.flow.V) > 0
        # Test that non-uniform V doesn't break
        sim = Simulation(nm,(0,0),radius; U=1, body=AutoBody(plate,rotate), ν, T, mem, exitBC)
        sim_step!(sim)
        @test sim.pois.n == [2,1]
        @test 1 > sim.flow.Δt[end] > 0.5
        # Test that divergent V doesn't break
        sim = Simulation(nm,(0,0),radius; U=1, body=AutoBody(plate,bend), ν, T, mem, exitBC)
        sim_step!(sim)
        @test sim.pois.n == [2,1]
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
        sim_step!(sim,1); write!(wr, sim); close(wr)

        # re start the sim from a paraview file
        restart = sphere_sim(;D,mem);
        restart_sim!(restart;fname="test_vtk_reader_$D.pvd")

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
using MPI
@testset "MPIArray.jl" begin
    buff = Base.get_extension(WaterLily, :WaterLilyMPIExt).buff
    halos = Base.get_extension(WaterLily, :WaterLilyMPIExt).halos
    copyto! = Base.get_extension(WaterLily, :WaterLilyMPIExt).copyto!
    fill_send! = Base.get_extension(WaterLily, :WaterLilyMPIExt).fill_send!
    for N ∈ [(16,8)] # not yet 3D
        for T in [Float32]
            a = zeros(T,N) |> MPIArray; a .= 1.0
            @test isa(a, MPIArray) && all(a .== 1.0)
            @test length(a.send[1])==length(a.send[2])==(length(N)-1)*2prod(ntuple(i->N[i],length(N)-1))
            a .= reshape(collect(1:prod(N)),N)
            a.send[1] .= 0; a.send[2] .= 0
            b = copy(a)
            @test all(a .== b) && size(a) == N && length(a) == prod(N) && ndims(a) == length(N) &&
                  eltype(a) == T && eltype(a.send[1]) == T && eltype(a.send[2]) == T
            # test halo filling
            fill_send!(a,1,Val(:Scalar))
            # check that they contain the right things
            @test all(reshape(a.send[1][1:2N[2]],(2,:)) .≈ a[buff(N,-1)]) # left
            @test all(reshape(a.send[2][1:2N[2]],(2,:)) .≈ a[buff(N,+1)]) # right
            fill_send!(a,2,Val(:Scalar)) # same but fill in the other way
            @test all(reshape(a.send[1][1:2N[1]],(:,2)) .≈ a[buff(N,-2)]) # bottom
            @test all(reshape(a.send[2][1:2N[1]],(:,2)) .≈ a[buff(N,+2)]) # top
            # reset the recv halos
            a.recv[1] .= 888.; a.recv[2] .= 999.
            copyto!(a,-1,Val(:Scalar))
            copyto!(a,+1,Val(:Scalar))
            @test all(a[halos(N,-1)] .≈ 888.) && all(a[halos(N,+1)] .≈ 999.)
            copyto!(a,-2,Val(:Scalar))
            copyto!(a,+2,Val(:Scalar))
            @test all(a[halos(N,-2)] .≈ 888.) && all(a[halos(N,+2)] .≈ 999.)
            # vector test
            v = zeros(T,(N...,length(N))) |> MPIArray; v.send[1] .= 0; v.send[2] .= 0
            v[:,:,1] .= reshape(collect(1:prod(N)),N)
            v[:,:,2] .= reshape(collect(prod(N):-1:1),N)
            fill_send!(v,1,Val(:Vector))
            @test all(reshape(v.send[1][1:4N[2]],(2,N[2],2)) .≈ v[buff(N,-1),:]) # left
            @test all(reshape(v.send[2][1:4N[2]],(2,N[2],2)) .≈ v[buff(N,+1),:]) # left
            fill_send!(v,2,Val(:Vector))
            @test all(reshape(v.send[1][1:4N[1]],(N[1],2,2)) .≈ v[buff(N,-2),:]) # bottom
            @test all(reshape(v.send[2][1:4N[1]],(N[1],2,2)) .≈ v[buff(N,+2),:]) # top
        end
    end
end
@testset "MPIExt.jl" begin
    Np = 2  # number of processes DO OT CHANGE
    run(`$(mpiexec()) -n $Np $(Base.julia_cmd()) --project=../ mpi_test.jl`)
    # load the results
    for n ∈ 0:Np-1
        g_loc = load("global_loc_$n.jld2")["data"]
        n==0 && @test all(g_loc[1] .≈ 0.5)
        n==0 && @test all(g_loc[2] .≈ SA[0.,0.5])
        n==1 && @test all(g_loc[1] .≈ SA[64.5,0.5])
        n==1 && @test all(g_loc[2] .≈ SA[64.0,0.5])
    
        # test that each one has it's rank as the data
        data = load("sigma_1_$n.jld2")["data"]
        @test all(data[3:end-2,3:end-2] .≈ n)
        
        # test halo swap
        data = load("sigma_2_$n.jld2")["data"]
        @test all(data[3:end-2,3:end-2] .≈ n)
        if n==0 # test that the boundaries are updated
            @test all(isnan.(data[1:2,4:end-2]))
            @test all(data[end-1:end,4:end-2] .≈ 1) 
        else
            @test all(isnan.(data[end-1:end,4:end-2]))
            @test all(data[1:2,4:end-2] .≈ 0) 
        end
        @test all(isnan.(data[3:end-1,1:2])) # this boundaries should not be updated
        @test all(isnan.(data[3:end-1,end-1:end]))
        
        # check the sdf in the two parts
        data = load("sdf_3_$n.jld2")["data"]
        a = copy(data); a .= 0.; L = 2^6
        f(x)=√sum(abs2,g_loc[1].-0.5+x.-SA[L/2,L/2+2])-L/8
        @inline lloc(i,I::CartesianIndex{N},T=Float32) where N = SVector{N,T}(I.I .- 2.5 .- 0.5 .* δ(i,I).I)
        @WaterLily.loop a[I] = f(lloc(0,I,eltype(a))) over I ∈ CartesianIndices(a)
        @test all(data[3:end-2,3:end-2] .≈ a[3:end-2,3:end-2])
    
        # check that halos are gathered correctly
        data = load("sdf_4_$n.jld2")["data"]
        @test all(data[3:end-2,3:end-2] .≈ a[3:end-2,3:end-2])
        if n==0
            @test all(isnan.(data[1:2,4:end-2]))
            @test all(data[end-1:end,4:end-2] .≈ a[end-1:end,4:end-2])
        else
            @test all(isnan.(data[end-1:end,4:end-2]))
            @test all(data[1:2,4:end-2] .≈ a[1:2,4:end-2]) 
        end
        # check that the norm are gathered correctly
        sol = 123.456789 # a not so random number
        data = load("norm_$n.jld2")["data"]
        @test all(data .≈ SA[sol,sol^2]) # L2 norm is square of Linf
    end

    # test pressure solver
    Np = 4
    run(`$(mpiexec()) -n $Np $(Base.julia_cmd()) --project=../ mpi_psolver_test.jl`)
    # test poisson solver in parallel
    for Pois in [:Poisson,:MultiLevelPoisson]
        for n ∈ 0:Np-1
            data = load("test_$(string(Pois))_$(n)_L2.jld2","data")
            @test all(data .≤ 2e-2)
        end
    end

    # clean the files
    @test_nowarn foreach(rm, filter(endswith(".jld2"), readdir())) 
end
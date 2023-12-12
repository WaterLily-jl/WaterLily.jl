using WaterLily
using Test
using CUDA: CUDA, @allowscalar
using AMDGPU: AMDGPU

function setup_backends()
    arrays = [Array]
    if CUDA.functional()
        CUDA.allowscalar(false)
        push!(arrays, CUDA.CuArray)
    end
    if AMDGPU.functional()
        AMDGPU.allowscalar(false)
        push!(arrays, AMDGPU.ROCArray)
    end
    return arrays
end

arrays = setup_backends()

@testset "util.jl" begin
    I = CartesianIndex(1,2,3,4)
    @test I+δ(3,I) == CartesianIndex(1,2,4,4)

    using StaticArrays
    @test loc(3,CartesianIndex(3,4,5)) == SVector(3,4,4.5) .- 1.5
    I = CartesianIndex(rand(2:10,3)...)
    @test loc(0,I) == SVector(I.I...) .- 1.5

    ex,sym = :(a[I,i] = Math.add(p.b[I],func(I,q))),[]
    WaterLily.grab!(sym,ex)
    @test ex == :(a[I, i] = Math.add(b[I], func(I, q)))
    @test sym == [:a, :I, :i, :(p.b), :q]

    # for f ∈ arrays
    for f ∈ [Array]
        p = Float64[i+j  for i ∈ 1:4, j ∈ 1:5] |> f
        @test inside(p) == CartesianIndices((2:3,2:4))
        @test inside(p,buff=0) == CartesianIndices(p)
        @test L₂(p) == 187

        u = zeros(5,5,2) |> f
        apply!((i,x)->x[i],u)
        @allowscalar @test [u[i,j,1].-(i-2) for i in 1:3, j in 1:3]==zeros(3,3)

        Ng, D, U = (6, 6), 2, (1.0, 0.5)
        u = rand(Ng..., D) |> f # vector
        σ = rand(Ng...) |> f # scalar
        BC!(u, U)
        BC!(σ)
        @allowscalar @test all(u[1, :, 1] .== U[1]) && all(u[2, :, 1] .== U[1]) && all(u[end, :, 1] .== U[1]) &&
                all(u[3:end-1, 1, 1] .== u[3:end-1, 2, 1]) && all(u[3:end-1, end, 1] .== u[3:end-1, end-1, 1])
        @allowscalar @test all(u[:, 1, 2] .== U[2]) && all(u[:, 2, 2] .== U[2]) && all(u[:, end, 2] .== U[2]) &&
                all(u[1, 3:end-1, 2] .== u[2, 3:end-1, 2]) && all(u[end, 3:end-1, 2] .== u[end-1, 3:end-1, 2])
        @allowscalar @test all(σ[1, 2:end-1] .== σ[2, 2:end-1]) && all(σ[end, 2:end-1] .== σ[end-1, 2:end-1]) &&
                all(σ[2:end-1, 1] .== σ[2:end-1, 2]) && all(σ[2:end-1, end] .== σ[2:end-1, end-1])

        @allowscalar u[end,:,1] .= 3
        BC!(u, U, true) # save exit values
        @allowscalar @test all(u[end, :, 1] .== 3)

        WaterLily.exitBC!(u,u,U,0) # conservative exit check
        @allowscalar @test all(u[end,2:end-1, 1] .== U[1])
    end
end

function Poisson_setup(poisson,N::NTuple{D};f=Array,T=Float32) where D
    c = ones(T,N...,D) |> f; BC!(c, ntuple(zero,D))
    x = zeros(T,N) |> f; z = copy(x)
    pois = poisson(x,c,z)
    soln = map(I->T(I.I[1]),CartesianIndices(N)) |> f
    I = first(inside(x))
    @allowscalar @. soln -= soln[I]
    z = mult!(pois,soln)
    solver!(pois)
    @allowscalar @. x -= x[I]
    return L₂(x-soln)/L₂(soln),pois
end

@testset "Poisson.jl" begin
    for f ∈ arrays
        err,pois = Poisson_setup(Poisson,(5,5);f)
        @test @allowscalar parent(pois.D)==f(Float32[0 0 0 0 0; 0 -2 -3 -2 0; 0 -3 -4 -3 0;  0 -2 -3 -2 0; 0 0 0 0 0])
        @test @allowscalar parent(pois.iD)≈f(Float32[0 0 0 0 0; 0 -1/2 -1/3 -1/2 0; 0 -1/3 -1/4 -1/3 0;  0 -1/2 -1/3 -1/2 0; 0 0 0 0 0])
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
        @test pois.n[] < 3
        err,pois = Poisson_setup(MultiLevelPoisson,(2^4+2,2^4+2,2^4+2);f)
        @test err < 1e-6
        @test pois.n[] < 3
    end
end

@testset "Flow.jl" begin
    # Check QUICK scheme on boundary
    ϕuL = WaterLily.ϕuL
    ϕuR = WaterLily.ϕuR
    quick = WaterLily.quick
    ϕ = WaterLily.ϕ
    
    # inlet with positive flux -> CD
    @test ϕuL(1,CartesianIndex(2),[0.,0.5,2.],1)==ϕ(1,CartesianIndex(2),[0.,0.5,2.0])
    # inlet negative flux -> backward QUICK
    @test ϕuL(1,CartesianIndex(2),[0.,0.5,2.],-1)==-quick(2.0,0.5,0.0)
    # outlet, positive flux -> standard QUICK
    @test ϕuR(1,CartesianIndex(3),[0.,0.5,2.],1)==quick(0.0,0.5,2.0)
    # outlet, negative flux -> backward CD
    @test ϕuR(1,CartesianIndex(3),[0.,0.5,2.],-1)==-ϕ(1,CartesianIndex(3),[0.,0.5,2.0])

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

    #test booleans
    @test all(measure(body1+body2,[-√2.,-√2.],1.).≈(-√2.,[-√.5,-√.5],[-2.,-2.]))
    @test all(measure(body1-body2,[-√2.,-√2.],1.).≈(√2.,[√.5,√.5],[-2.,-2.]))
end

using StaticArrays
function get_flow(N,f)
    a = Flow((N,N),(1.,0.);f,T=Float32)
    @inside a.p[I] = loc(0, I)[2]
    sdf(x,t) = √sum(abs2,x.-(N÷2+0.5))-N÷4
    map(x,t) = x.-SVector(t,0)
    body = AutoBody(sdf,map)
    WaterLily.measure!(a,body)
    return a,body
end

@testset "Flow.jl with Body.jl" begin
    # Horizontally moving body
    for f ∈ arrays
        a,_ = get_flow(20,f)
        mom_step!(a,Poisson(a.p,a.μ₀,a.σ))
        @test mapreduce(abs2,+,a.u[:,5,1].-1) < 6e-5
    end
end
import WaterLily: ×
@testset "Metrics.jl" begin
    J = CartesianIndex(2,3,4); x = loc(0,J); px = prod(x)
    for f ∈ arrays
        u = zeros(3,4,5,3) |> f; apply!((i,x)->x[i]+prod(x),u)
        p = zeros(3,4,5) |> f
        @inside p[I] = WaterLily.ke(I,u)
        @test @allowscalar p[J]==0.5*sum(abs2,x .+ px)
        @inside p[I] = WaterLily.ke(I,u,x)
        @test @allowscalar p[J]==1.5*px^2
        @inside p[I] = WaterLily.λ₂(I,u)
        @test @allowscalar p[J]≈1
        ω = (1 ./ x)×repeat([px],3)
        @inside p[I] = WaterLily.curl(2,I,u)
        @test @allowscalar p[J]==ω[2]
        f==Array && @test WaterLily.ω(J,u)≈ω
        @inside p[I] = WaterLily.ω_mag(I,u)
        @test @allowscalar p[J]==sqrt(sum(abs2,ω))
        @inside p[I] = WaterLily.ω_θ(I,(0,0,1),x .+ (0,1,2),u)
        @test @allowscalar p[J]≈ω[1]

        N = 32
        a,body = get_flow(N,f)
        force = WaterLily.∮nds(a.p,a.V,body)
        @show force/(π*(N÷4)^2)- [0,1]
        @test sum(abs,force/(π*(N÷4)^2) - [0,1]) < 1e-2
    end
end

function sphere_sim(radius = 8; mem=Array, exitBC=false)
    body = AutoBody((x,t)-> √sum(abs2,x .- (2radius+1.5)) - radius)
    return Simulation(radius.*(6,4),(1,0),radius; body, ν=radius/250, T=Float32, mem, exitBC)
end
@testset "WaterLily.jl" begin
    for mem ∈ arrays, exitBC ∈ (true,false)
        sim = sphere_sim(;mem,exitBC);
        @test sim_time(sim) == 0
        sim_step!(sim,0.1,remeasure=false)
        @test length(sim.flow.Δt)-1 == length(sim.pois.n)÷2
    end
end
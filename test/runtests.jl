using WaterLily
using Test
using PerformanceTestTools
using JLD2

@testset "multithreaded equivalence" begin
    N_multithread = 2
    PerformanceTestTools.@include_foreach(
        "tests_using_threads.jl",
        [["JULIA_NUM_THREADS" => "1"]],
    )
    PerformanceTestTools.@include_foreach(
        "tests_using_threads.jl",
        [["JULIA_NUM_THREADS" => string(N_multithread)]],
    )
    filebase = string(tempdir(), "/testing_using_threads#")
    sim1_u = load(string(filebase, 1, "_u.jld2"))["data"]
    sim2_u = load(string(filebase, N_multithread, "_u.jld2"))["data"]
    @test maximum(broadcast(abs, sim1_u-sim2_u)) < 1e-3
end

@testset "util.jl" begin
    @test L₂(2ones(4,4)) == 16
    a = Float64[i+j+k  for i ∈ 1:4, j ∈ 1:4, k ∈ 1:2]
    BC!(a,[0. 0.])
    @test a == cat([0. 0. 0. 0.
                    0. 0. 0. 0.
                    6. 6. 7. 7.
                    0. 0. 0. 0.],
                   [0. 0. 7. 0.
                    0. 0. 7. 0.
                    0. 0. 8. 0.
                    0. 0. 8. 0.],dims=3)
    u = zeros(4,4,2); apply!((i,x)->x[i],u)
    @test [u[i,j,1].-(i-0.5) for i in 1:4, j in 1:4]==zeros(4,4)
    using StaticArrays
    @test loc(3,CartesianIndex(3,4,5)) == SVector(3,4,4.5)
    I = CartesianIndex(rand(2:10,3)...)
    @test loc(0,I) == SVector(I.I...)
end

function Poisson_test_2D(f,n)
    c = ones(n+2,n+2,2); BC!(c,[0. 0.])
    p = f(c)
    soln = Float64[ i for i ∈ 1:n+2, j ∈ 1:n+2]
    b = mult(p,soln)
    x = zeros(n+2,n+2)
    solver!(x,p,b)
    x .-= (x[2,2]-soln[2,2])
    return L₂(x.-soln)/L₂(soln)
end
function Poisson_test_3D(f,n)
    c = ones(n+2,n+2,n+2,3); BC!(c,[0. 0. 0.])
    p = f(c)
    soln = Float64[ i for i ∈ 1:n+2, j ∈ 1:n+2, k ∈ 1:n+2]
    b = mult(p,soln)
    x = zeros(n+2,n+2,n+2)
    solver!(x,p,b,tol=1e-5)
    x .-= (x[2,2,2]-soln[2,2,2])
    return L₂(x.-soln)/L₂(soln)
end

@testset "Poisson.jl" begin
    @test Poisson_test_2D(Poisson,2^6) < 1e-5
    @test Poisson_test_3D(Poisson,2^4) < 1e-5
end
@testset "MultiLevelPoisson.jl" begin
    I = CartesianIndex(4,3,2)
    @test all(WaterLily.down(J)==I for J ∈ WaterLily.up(I))
    @test_throws AssertionError("MultiLevelPoisson requires size=a2ⁿ, where a<31, n>2") Poisson_test_2D(MultiLevelPoisson,67)
    @test_throws AssertionError("MultiLevelPoisson requires size=a2ⁿ, where a<31, n>2") Poisson_test_3D(MultiLevelPoisson,3^4)
    @test Poisson_test_2D(MultiLevelPoisson,2^6) < 1e-5
    @test Poisson_test_3D(MultiLevelPoisson,2^4) < 1e-5
end

@testset "Body.jl" begin
    @test WaterLily.μ₀(3,6)==WaterLily.μ₀(0.5,1)
    @test WaterLily.μ₀(0,1)==0.5
    @test WaterLily.μ₁(0,2)==2*(1/4-1/π^2)
end

@testset "AutoBody.jl" begin
    using LinearAlgebra: norm2
    body = AutoBody((x,t)->norm2(x)-2-t)
    @test all(measure(body,[√2.,√2.],0.).≈(0,[√.5,√.5],[0.,0.]))
    @test all(measure(body,[2.,0.,0.],1.).≈(-1.,[1.,0.,0.],[0.,0.,0.]))
    body = AutoBody((x,t)->norm2(x)-2,(x,t)->x.+t^2)
    @test all(measure(body,[√2.,√2.],0.).≈(0,[√.5,√.5],[0.,0.]))
    @test all(WaterLily.measure(body,[1.,-1.,-1.],1.).≈(0.,[1.,0.,0.],[-2.,-2.,-2.]))
    dims = (2^5,2^5)
    sdf(x) = norm2(x.-2^4)-4π
    a = zeros(dims); WaterLily.apply_sdf!(sdf,a)
    b = zeros(dims); @inside b[I] = sdf(WaterLily.loc(0,I))
    @test all(@. (a==b) | (abs(b)>2))
end

@testset "Flow.jl" begin
    # Horizontally moving body
    using LinearAlgebra: norm2
    a = Flow((20,20),[1.,0.])
    center = [10.58,10.65] # worst case - not sure why
    measure!(a,AutoBody((x,t)->norm2(x.-center)-5,(x,t)->x.-[t,0.]))
    mom_step!(a,Poisson(a.μ₀))
    @test sum(abs2,a.u[:,5,1].-1) < 2e-5

    # Impulsive flow in a box
    U = [2/3,-1/3]
    a = Flow((14,10),U)
    mom_step!(a,MultiLevelPoisson(a.μ₀))
    @test L₂(a.u[:,:,1].-U[1]) < 2e-5
    @test L₂(a.u[:,:,2].-U[2]) < 1e-5
end

@testset "Metrics.jl" begin
    I = CartesianIndex(2,3,4)
    u = zeros(3,4,5,3); apply!((i,x)->x[i]+prod(x),u)
    @test WaterLily.ke(I,u)==0.5*(26^2+27^2+28^2)
    @test WaterLily.ke(I,u,[2,3,4])===1.5*24^2
    @test [WaterLily.∂(i,j,I,u)
            for i in 1:3, j in 1:3] == [13 8 6; 12 9 6; 12 8 7]
    @test WaterLily.λ₂(I,u)≈1
    ω = [8-6,6-12,12-8]
    @test WaterLily.curl(2,I,u)==ω[2]
    @test WaterLily.ω(I,u)==ω
    @test WaterLily.ω_mag(I,u)==sqrt(sum(abs2,ω))
    @test WaterLily.ω_θ(I,[0,0,1],[2,2,2],u)==-ω[1]

    body = AutoBody((x,t)->√sum(abs2,x .- 2^6) - 2^5)
    p = ones(2^7,2^7)
    @inside p[I] = sum(I.I[2])
    @test sum(abs2,WaterLily.∮nds(p,body)/(π*2^10).-(0,1))<1e-6
    @inside p[I] = cos(atan(reverse(loc(0,I) .- 2^6)...))
    @test sum(abs2,WaterLily.∮nds(p,body)/(π*2^5).-(1,0))<1e-6
end

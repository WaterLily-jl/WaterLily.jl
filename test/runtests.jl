using WaterLily
using Test

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
end

function Poisson_test_2D(f,n)
    c = ones(2^n+2,2^n+2,2); BC!(c,[0. 0.])
    p = f(c)
    soln = Float64[ i for i ∈ 1:2^n+2, j ∈ 1:2^n+2]
    b = mult(p,soln)
    x = zeros(2^n+2,2^n+2)
    solve!(x,p,b)
    x .-= (x[2,2]-soln[2,2])
    return L₂(x.-soln)/L₂(soln)
end
function Poisson_test_3D(f,n)
    c = ones(2^n+2,2^n+2,2^n+2,3); BC!(c,[0. 0. 0.])
    p = f(c)
    soln = Float64[ i for i ∈ 1:2^n+2, j ∈ 1:2^n+2, k ∈ 1:2^n+2]
    b = mult(p,soln)
    x = zeros(2^n+2,2^n+2,2^n+2)
    solve!(x,p,b)
    x .-= (x[2,2,2]-soln[2,2,2])
    return L₂(x.-soln)/L₂(soln)
end

@testset "Poisson.jl" begin
    @test Poisson_test_2D(Poisson,6) < 1e-5
    @test Poisson_test_3D(Poisson,4) < 1e-5
end
@testset "MultiLevelPoisson.jl" begin
    @test Poisson_test_2D(MultiLevelPoisson,6) < 1e-5
    @test Poisson_test_3D(MultiLevelPoisson,4) < 1e-5
end

@testset "Body.jl" begin
    u = apply((i,x)->x[i],4,4,2)
    @test [u[i,j,1].-(i-0.5) for i in 1:4, j in 1:4]==zeros(4,4)
    @test [u[i,j,1].-(i-0.5) for i in 1:4, j in 1:4]==zeros(4,4)
    @test BDIM_coef(i->1,4,8,2)==ones(4,8,2)
    @test BDIM_coef(i->0,4,8,2)==0.5ones(4,8,2)
    @test BDIM_coef(i->-1,4,8,2)≈zeros(4,8,2) atol=2eps(1.)
end

@testset "Flow.jl" begin
    # Impulsive flow in a box
    u = zeros(6,10,2)
    c = ones(6,10,2)
    U = [2/3,-1/3]
    a = Flow(u,c,U)
    b = MultiLevelPoisson(c)
    mom_step!(a,b) # now they should match
    @test L₂(a.u[:,:,1].-U[1]) < 1e-6*4*8
    @test L₂(a.u[:,:,2].-U[2]) < 1e-6*4*8
end

@testset "Metrics.jl" begin
    u = apply((i,x)->x[i]+prod(x),3,4,5,3)
    @test WaterLily.ke(CartesianIndex(2,3,4),u)==0.5*(26^2+27^2+28^2)
    @test WaterLily.ke(CartesianIndex(2,3,4),u,[2,3,4])===1.5*24^2
    @test [WaterLily.∂(i,j,CartesianIndex(2,3,4),u)
            for i in 1:3, j in 1:3] == [13 8 6; 12 9 6; 12 8 7]
    @test WaterLily.λ₂(CartesianIndex(2,3,4),u)==1
    ω = [8-6,6-12,12-8]
    @test WaterLily.curl(2,CartesianIndex(2,3,4),u)==ω[2]
    @test WaterLily.ω(CartesianIndex(2,3,4),u)==ω
    @test WaterLily.ω_mag(CartesianIndex(2,3,4),u)==sqrt(sum(abs2,ω))
    @test WaterLily.ω_θ(CartesianIndex(2,3,4),[0,0,1],[2,2,2],u)==-ω[1]
end

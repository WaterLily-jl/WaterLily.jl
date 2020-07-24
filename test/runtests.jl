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
    c = ones(4,8,2); BC!(c,zeros(2))
    @test BDIM_coef(i->1,4,8,2) ≈ c
    @test L₂(BDIM_coef(i->-1,4,8,2))==0.0
end

@testset "Flow.jl" begin
    # Impulsive flow in a box with mismatched BCs
    U = rand(2)
    u = zeros(6,10,2); BC!(u,U) # u≠U
    c = ones(6,10,2); BC!(c,[0.,0.])
    a = Flow(u,c,U)
    b = MultiLevelPoisson(c)
    mom_step!(a,b) # now they should match
    @test L₂(a.u[:,:,1].-U[1]) < 1e-6
    @test L₂(a.u[:,:,2].-U[2]) < 1e-6
end

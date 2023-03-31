using WaterLily
using Test
using BenchmarkTools
using OffsetArrays
OA(D=0) = OffsetArrays.Origin(D > 0 ? (zeros(Int, D)..., 1) : 0)
function test_Array(f,n)
    c = ones(n+2,n+2,2); BC!(c,[0. 0.])
    x = zeros(n+2,n+2)
    p = f(x,c)
    soln = Float64[ i for i ∈ 1:n+2, j ∈ 1:n+2]
    b = mult(p,soln)
    @btime WaterLily.residual!($p,$b)
    @btime solver!($p,$b) evals=1
    x .-= (x[2,2]-soln[2,2])
    return L₂(x.-soln)/L₂(soln)
end
function test_OA(f,n)
    c = ones(n+2,n+2,2)|>OA(2)
    c[0,:,1] .= c[1,:,1] .= c[end,:,1] .= 0 
    c[:,0,2] .= c[:,1,2] .= c[:,end,2] .= 0
    x = zeros(n+2,n+2)|>OA()
    p = f(x,c)
    soln = Float64[ i for i ∈ 1:n+2, j ∈ 1:n+2]|>OA()
    b = mult(p,soln)
    @btime WaterLily.residual!($p,$b)
    @btime solver!($p,$b) evals=1
    x .-= (x[2,2]-soln[2,2])
    return L₂(x.-soln)/L₂(soln)
end
test_Array(Poisson,2^6)
test_OA(Poisson,2^6)

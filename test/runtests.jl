using WaterLily
using Test
using CUDA: cu, @allowscalar, allowscalar
allowscalar(false)

@testset "util.jl" begin
    I = CartesianIndex(1,2,3,4)
    @test I+δ(3,I) == CartesianIndex(1,2,4,4)

    p = Float64[i+j  for i ∈ 1:4, j ∈ 1:5]
    @test inside(p) == CartesianIndices((2:3,2:4))
    @test L₂(p) == 187

    p = p |> OA()
    @test inside(p) == CartesianIndices((1:2,1:3))
    @test L₂(p) == 187 # unchanged!

    p = p |> cu
    @test L₂(p) == 187 # unchanged!

    u = Float64[i+j+k  for i ∈ 1:4, j ∈ 1:4, k ∈ 1:2] |> OA(2)
    @test first.(axes(u)) == (0,0,1)

    using StaticArrays
    @test loc(3,CartesianIndex(3,4,5)) == SVector(3,4,4.5)
    I = CartesianIndex(rand(2:10,3)...)
    @test loc(0,I) == SVector(I.I...)

    ex,sym = :(a[I,i] = Math.add(p.b[I],func(I,q))),[]
    WaterLily.grab!(sym,ex)
    @test ex == :(a[I, i] = Math.add(b[I], func(I, q)))
    @test sym == [:a, :I, :i, :(p.b), :q]

    for f ∈ [identity, cu]
        u = zeros(5,5,2) |> OA(2) |> f
        apply!((i,x)->x[i],u)
        @allowscalar @test [u[i,j,1].-(i-0.5) for i in 1:3, j in 1:3]==zeros(3,3)

        Ng, D, U = (6, 6), 2, (1.0, 0.5)
        u = rand(Ng..., D) |> OA(D) |> f # vector
        σ = rand(Ng...) |> OA() |> f  # scalar
        bc = WaterLily.bc_indices(Ng) |> f # bcs list
        BC!(u, U, bc)
        BC!(σ, bc)
        allowscalar() do
            @test all(u[0, :, 1] .== U[1]) && all(u[1, :, 1] .== U[1]) &&
                all(u[3:end-1, 0, 1] .== u[3:end-1, 1, 1]) && all(u[3:end-1, end, 1] .== u[3:end-1, end, 1])
            @test all(u[:, 0, 2] .== U[2]) && all(u[:, 1, 2] .== U[2]) &&
                all(u[0, 3:end-1, 2] .== u[1, 3:end-1, 2]) && all(u[end, 3:end-1, 2] .== u[end, 3:end-1, 2])
            @test all(σ[0, 1:end-1] .== σ[1, 1:end-1]) && all(σ[end, 1:end-1] .== σ[end-1, 1:end-1]) &&
                all(σ[1:end-1, 0] .== σ[1:end-1, 0]) && all(σ[1:end-1, end] .== σ[1:end-1, end-1])
        end
    end
end

function Poisson_setup(poisson,N;f=identity,T=Float32,D=length(N))
    c = ones(T,N...,D) |> f|> OA(D)
    @allowscalar c[0,:,1] .= c[1,:,1] .= c[end,:,1] .= 0 # to be substituted by BC!
    @allowscalar c[:,0,2] .= c[:,1,2] .= c[:,end,2] .= 0 # to be substituted by BC!
    x = zeros(T,N) |> f |> OA()
    pois = poisson(x,c)
    soln = map(I->T(I.I[1]),CartesianIndices(N)) |> f |> OA()
    solver!(pois,mult(pois,soln))
    @allowscalar @. x -= soln+(x[2,2]-soln[2,2])
    return L₂(x)/L₂(soln),pois
end

@testset "Poisson.jl" begin
    for f ∈ [identity,cu]
        err,pois = Poisson_setup(Poisson,(5,5);f)
        @test @allowscalar parent(pois.D)==f(Float32[0 0 0 0 0; 0 -2 -3 -2 0; 0 -3 -4 -3 0;  0 -2 -3 -2 0; 0 0 0 0 0])
        @test @allowscalar parent(pois.iD)≈f(Float32[0 0 0 0 0; 0 -1/2 -1/3 -1/2 0; 0 -1/3 -1/4 -1/3 0;  0 -1/2 -1/3 -1/2 0; 0 0 0 0 0])
        @test err < 1e-6
        err,_ = Poisson_setup(Poisson,(2^6+2,2^6+2))
        @test err < 1e-5
    end
end

@testset "MultiLevelPoisson.jl" begin
    I = CartesianIndex(4,3,2)
    @test all(WaterLily.down(J)==I for J ∈ WaterLily.up(I))
    @test_throws AssertionError("MultiLevelPoisson requires size=a2ⁿ, where a<31, n>2") Poisson_setup(MultiLevelPoisson,(15+2,3^4+2))
    err,pois = Poisson_setup(MultiLevelPoisson,(10,10))
    @test L₂(pois.levels[2].D) == 4sum(abs2,[-2,-3,-3,-4])
    @test err < 1e-5
    # err,pois = Poisson_setup(MultiLevelPoisson,(2^6+2,2^6+2))
    # @show pois.levels |> length
    # @show pois.n
    # @test err < 1e-5
end

# @testset "Body.jl" begin
#     @test WaterLily.μ₀(3,6)==WaterLily.μ₀(0.5,1)
#     @test WaterLily.μ₀(0,1)==0.5
#     @test WaterLily.μ₁(0,2)==2*(1/4-1/π^2)
# end

# @testset "AutoBody.jl" begin
#     using LinearAlgebra: norm2

#     # test AutoDiff in 2D and 3D
#     body1 = AutoBody((x,t)->norm2(x)-2-t)
#     @test all(measure(body1,[√2.,√2.],0.).≈(0,[√.5,√.5],[0.,0.]))
#     @test all(measure(body1,[2.,0.,0.],1.).≈(-1.,[1.,0.,0.],[0.,0.,0.]))
#     body2 = AutoBody((x,t)->norm2(x)-2,(x,t)->x.+t^2)
#     @test all(measure(body2,[√2.,√2.],0.).≈(0,[√.5,√.5],[0.,0.]))
#     @test all(measure(body2,[1.,-1.,-1.],1.).≈(0.,[1.,0.,0.],[-2.,-2.,-2.]))

#     #test booleans
#     @test all(measure(body1+body2,[-√2.,-√2.],1.).≈(-√2.,[-√.5,-√.5],[-2.,-2.]))
#     @test all(measure(body1-body2,[-√2.,-√2.],1.).≈(√2.,[√.5,√.5],[-2.,-2.]))

#     # test fast apply_sdf matches exhaustive sdf
#     dims = (2^5,2^5)
#     sdf(x) = norm2(x.-2^4)-4π
#     a = zeros(dims); WaterLily.apply_sdf!(sdf,a)
#     b = zeros(dims); @inside b[I] = sdf(WaterLily.loc(0,I))
#     @test all(@. clamp(a,-2,2)==clamp(b,-2,2))
# end

# @testset "Flow.jl" begin
#     # Horizontally moving body
#     using LinearAlgebra: norm2
#     a = Flow((20,20),[1.,0.])
#     center = [10.58,10.65] # worst case - not sure why
#     measure!(a,AutoBody((x,t)->norm2(x.-center)-5,(x,t)->x.-[t,0.]))
#     mom_step!(a,Poisson(a.μ₀))
#     @test sum(abs2,a.u[:,5,1].-1) < 2e-5

#     # Impulsive flow in a box
#     U = [2/3,-1/3]
#     a = Flow((14,10),U)
#     mom_step!(a,MultiLevelPoisson(a.μ₀))
#     @test L₂(a.u[:,:,1].-U[1]) < 2e-5
#     @test L₂(a.u[:,:,2].-U[2]) < 1e-5
# end

# @testset "Metrics.jl" begin
#     I = CartesianIndex(2,3,4)
#     u = zeros(3,4,5,3); apply!((i,x)->x[i]+prod(x),u)
#     @test WaterLily.ke(I,u)==0.5*(26^2+27^2+28^2)
#     @test WaterLily.ke(I,u,[2,3,4])===1.5*24^2
#     @test [WaterLily.∂(i,j,I,u)
#             for i in 1:3, j in 1:3] == [13 8 6; 12 9 6; 12 8 7]
#     @test WaterLily.λ₂(I,u)≈1
#     ω = [8-6,6-12,12-8]
#     @test WaterLily.curl(2,I,u)==ω[2]
#     @test WaterLily.ω(I,u)==ω
#     @test WaterLily.ω_mag(I,u)==sqrt(sum(abs2,ω))
#     @test WaterLily.ω_θ(I,[0,0,1],[2,2,2],u)==-ω[1]

#     body = AutoBody((x,t)->√sum(abs2,x .- 2^6) - 2^5)
#     p = ones(2^7,2^7)
#     @inside p[I] = sum(I.I[2])
#     @test sum(abs2,WaterLily.∮nds(p,body)/(π*2^10).-(0,1))<1e-6
#     @inside p[I] = cos(atan(reverse(loc(0,I) .- 2^6)...))
#     @test sum(abs2,WaterLily.∮nds(p,body)/(π*2^5).-(1,0))<1e-6
# end

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

function poisson_test(n)
    c = ones(2^n+2,2^n+2,2); BC!(c,[0. 0.])
    p = PoissonSys(c)
    soln = Float64[ i for i ∈ 1:2^n+2, j ∈ 1:2^n+2]
    b = mult(p,soln)
    x = zeros(2^n+2,2^n+2)
    solve!(x,p,b)
    x .-= x[2,2]-2
    return L₂(x.-soln)/L₂(soln)
end

@testset "PoissonSys.jl" begin
    @test poisson_test(7) < 1e-5
end

function GMG_test(n)
    c = ones(2^n+2,2^n+2,2); BC!(c,[0. 0.])
    p = MultiLevelPS(c)
    soln = Float64[ i for i ∈ 1:2^n+2, j ∈ 1:2^n+2]
    b = mult(p,soln)
    x = zeros(2^n+2,2^n+2)
    solve!(x,p,b)
    x .-= x[2,2]-2
    return L₂(x.-soln)/L₂(soln)
end

@testset "GMG.jl" begin
    @test GMG_test(7) < 1e-5
end

mom_test(a::Flow,b::Poisson,n=1000) = @time for i ∈ 1:n
    mom_step!(a,b)
end

function TwoD_block(n,m;xr=1:0,yr=1:0,U=[1.,0.])
    u = zeros(n+2,m+2,2); BC!(u,U)
    c = ones(n+2,m+2,2); BC!(c,[0. 0.])

    # immerse a solid block (proto-BDIM)
    u[first(xr):last(xr)+1,yr,1] .= 0
    c[first(xr):last(xr)+1,yr,1] .= 0
    c[xr,first(yr):last(yr)+1,2] .= 0

    return Flow(u,c,U,Δt=0.1,ν=0.01)
end

function TwoD_block_test(p=7,N=[5000,1000])
    n,m = 2^p,2^(p-1); xr = m÷2:m÷2; yr = 3m÷8+2:5m÷8+1
    a = TwoD_block(n,m,xr=xr,yr=yr);
    b = MultiLevelPS(a.μ₀)
    for n ∈ N
        @show n
        mom_test(a,b,n)
    end
    # show(a.p,-3,1)
    # show([curl(3,I,a.u) for I ∈ inside(a.p)],-1,1)
    return a,b
end

function TGVortex(p)
    L,U = 2^p,zeros(3)
    u = [-sin((i-2)*π/L)*cos((j-1.5)*π/L)*cos((k-1.5)*π/L) for i∈1:L+2, j∈1:L+2, k∈1:L+2]
    v = [ cos((i-1.5)*π/L)*sin((j-2)*π/L)*cos((k-1.5)*π/L) for i∈1:L+2, j∈1:L+2, k∈1:L+2]
    u = cat(u,v,zeros(L+2,L+2,L+2),dims=4); BC!(u,U)
    c = ones(L+2,L+2,L+2,3); BC!(c,U)
    return Flow(u,c,U,Δt=0.1,ν=0.01)
end

function TGVortex_test(p=7,N=[1,10,100])
    a = TGVortex(p)
    b = MultiLevelPS(a.μ₀)
    for n ∈ N
        @show n
        mom_test(a,b,n)
    end
    # ω₃ = [curl(3,I,a.u) for I ∈ inside(a.p)]
    # show(ω₃[:,:,2^(p-2)],-0.25,0.25)
    return a,b
end
#--------------------------
# gr(show = false)
# @gif for time ∈ 0:2^(p+5),Δt=0.25
#     mom_step!(a,b,U=U,ν=0.01,Δt=0.25)
#     show(curl₃(@view a.u[:,:,2^(p-1),:]),-0.25,0.25)
# end

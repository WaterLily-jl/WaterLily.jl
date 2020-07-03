include("WaterLily.jl");include("GMG.jl")
using Profile,ProfileView
function mom_test(a::Flow,b::Poisson,U,n=1000)
    @time for i ∈ 1:n
        mom_step!(a,b,U=U,ν=0.01,Δt=0.1)
    end
end

function TwoD_block(n,m;xr=1:0,yr=1:0,U=[1. 0.])
    u = zeros(n+2,m+2,2); BC!(u,U)
    c = ones(n+2,m+2,2); BC!(c,[0. 0.])

    # immerse a solid block (proto-BDIM)
    u[first(xr):last(xr)+1,yr,1] .= 0
    c[first(xr):last(xr)+1,yr,1] .= 0
    c[xr,first(yr):last(yr)+1,2] .= 0

    return Flow(u,c),U
end

function TwoD_block_test(p=7,N=[5000,1000])
    n,m = 2^p,2^(p-1); xr = m÷2:m÷2; yr = 3m÷8+2:5m÷8+1
    a,U = TwoD_block(n,m,xr=xr,yr=yr);
    b = MultiLevelPS(a.c)
    for n ∈ N
        @show n
        mom_test(a,b,U,n)
    end
    show(a.p,-3,1)
end

function TGVortex(p)
    L = 2^p
    u = [-sin((i-2)*π/L)*cos((j-1.5)*π/L)*cos((k-1.5)*π/L) for i∈1:L+2, j∈1:L+2, k∈1:L+2]
    v = [ cos((i-1.5)*π/L)*sin((j-2)*π/L)*cos((k-1.5)*π/L) for i∈1:L+2, j∈1:L+2, k∈1:L+2]
    u = cat(u,v,zeros(L+2,L+2,L+2),dims=4); BC!(u,[0. 0. 0.])
    c = ones(L+2,L+2,L+2,3); BC!(c,[0. 0. 0.])
    return Flow(u,c),[0. 0. 0.]
end

function TGVortex_test(p=7,N=[1,10,100])
    a,U = TGVortex(p)
    b = MultiLevelPS(a.c)
    for n ∈ N
        @show n
        mom_test(a,b,U,n)
    end
    show(curl₃(@view a.u[:,:,2^(p-2),:]),-0.025,0.025)
end
#--------------------------
# gr(show = false)
# @gif for time ∈ 0:2^(p+5),Δt=0.25
#     mom_step!(a,b,U=U,ν=0.01,Δt=0.25)
#     show(curl₃(@view a.u[:,:,2^(p-1),:]),-0.25,0.25)
# end
#-------------------------------------------------
function poisson_test(n)
    c = ones(2^n+2,2^n+2,2); BC!(c,[0. 0.])
    p = PoissonSys(c)
    x = Float64[i  for i ∈ 1:2^n+2, j ∈ 1:2^n+2]
    b = mult(p,x)
    fill!(x,0.)
    @time solve!(x,p,b,log=true)
end
#
function GMG_test(n)
    c = ones(2^n+2,2^n+2,2); BC!(c,[0. 0.])
    @time p = MultiLevelPS(c)
    x = Float64[i for i∈1:2^n+2, j∈1:2^n+2]
    b = mult(p.levels[1],x)
    fill!(x,0.)
    @time solve!(x,p,b,log=true)
end
#-------------------------------------------------
function tracer_init(n,m)
    f = zeros(n+2,m+2)
    f[10:n÷4+10, m-n÷4-5:m-5] .= 1
    u = [ sin((i-2)*π/m)*cos((j-1.5)*π/m) for i ∈ 1:n+2, j ∈ 1:m+2 ]
    v = [-cos((i-1.5)*π/m)*sin((j-2)*π/m) for i ∈ 1:n+2, j ∈ 1:m+2 ]
    return f,similar(f),cat(u,v,dims=3)
end

function tracer_test(n=1000,Δt=0.25)
    p = 8; Pe = 2^(p-9.)/50
    f,r,u = tracer_init(2^p,2^(p-1))
    @time for time ∈ 0:n
        fill!(r,0.)
        tracer_transport!(r,f,u,Pe=Pe)
        @. f += Δt*r; BCᶜ!(f)
    end
end

# gr(show = false)
# @gif for time ∈ 0:2^(n+5),Δt=0.25
#     fill!(r,0.)
#     tracer_transport!(r,f,u,Pe=Pe)
#     @. f += Δt*r; BCᶜ!(f)
#     show(f)
# end every 2^(n-2)

function mom_init(n,m;xr=1:0,yr=1:0)
    uˣ = zeros(n+2,m+2); BCˣ!(uˣ,1.)
    uʸ = zeros(n+2,m+2)

    cˣ = ones(n+2,m+2); BCˣ!(cˣ,0.)
    cʸ = ones(n+2,m+2); BCʸ!(cʸ,0.)

    # immerse a solid block (proto-BDIM)
    cˣ[first(xr):last(xr)+1,yr] .= 0
    cʸ[xr,first(yr):last(yr)+1] .= 0

    p = zeros(n+2,m+2)
    rˣ = zeros(n+2,m+2)
    rʸ = zeros(n+2,m+2)
    σ = zeros(n*m)
    p_vec = zeros(n*m)

    return flow(uˣ,uʸ,cˣ,cʸ,rˣ,rʸ,p,MG(cˣ,cʸ),σ,p_vec)
end

n,m = 2^7,2^6; xr = m÷2:m÷2; yr = 3m÷8+2:5m÷8+1
a = mom_init(n,m,xr=xr,yr=yr);
mom_step!(a,ν=0.01,Δt=0.1)
a.p[xr,yr] .= -3.
show(a.p,-3,1)

using Profile,ProfileView
function test(n=1000)
    @time for i ∈ 1:n
        mom_step!(a,ν=0.01,Δt=0.1)
    end
end

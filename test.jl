function mom_init(n,m;xr=1:0,yr=1:0)
    p = zeros(n+2,m+2)
    uˣ = zeros(n+2,m+2); BCˣ!(uˣ,1.)
    uʸ = zeros(n+2,m+2)

    cˣ = ones(n+2,m+2); BCˣ!(cˣ,0.)
    cʸ = ones(n+2,m+2); BCʸ!(cʸ,0.)

    # immerse a solid block (proto-BDIM)
    cˣ[first(xr):last(xr)+1,yr] .= 0
    cʸ[xr,first(yr):last(yr)+1] .= 0

    return p,uˣ,uʸ,cˣ,cʸ,MG(cˣ,cʸ)
end

n,m = 128,64; r = 3m÷8+2:5m÷8+1
p,uˣ,uʸ,cˣ,cʸ,ml = mom_init(n,m,xr=m÷2:m÷2,yr=r);
# σ = zeros(n*m); x = zeros(n*m);
# rˣ = similar(uˣ); rʸ = similar(uʸ);
mom_step!(p,uˣ,uʸ,cˣ,cʸ,ml,ν=0.01,Δt=0.1)
show(p,-3,1)

function updatef()
    @time for i ∈ 1:1000
        mom_step!(p,uˣ,uʸ,cˣ,cʸ,ml,ν=0.01,Δt=0.1)
    end
end

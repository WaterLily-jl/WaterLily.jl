using WaterLily
using BenchmarkTools

function TwoD_block_test(p=7,N=[1,5000])
    n,m = 2^p,2^(p-1); xr = m÷2:m÷2; yr = 3m÷8+2:5m÷8+1
    u = zeros(n+2,m+2,2); BC!(u,[1.,0.])
    c = ones(n+2,m+2,2); BC!(c,[0. 0.])
    c[first(xr):last(xr)+1,yr,1] .= 0
    c[xr,first(yr):last(yr)+1,2] .= 0
    a = Flow(u,c,[1.,0.],Δt=0.1,ν=0.01)
    b = MultiLevelPS(c)
    for n ∈ N
        @show n
        for i ∈ 1:n
            mom_step!(a,b)
        end
        @btime mom_step!($a,$b)
    end
end

function TGVortex_test(p=7,N=[1,10])
    L,U = 2^p,zeros(3)
    u = [-sin((i-2)*π/L)*cos((j-1.5)*π/L)*cos((k-1.5)*π/L) for i∈1:L+2, j∈1:L+2, k∈1:L+2]
    v = [ cos((i-1.5)*π/L)*sin((j-2)*π/L)*cos((k-1.5)*π/L) for i∈1:L+2, j∈1:L+2, k∈1:L+2]
    u = cat(u,v,zeros(L+2,L+2,L+2),dims=4); BC!(u,U)
    c = ones(L+2,L+2,L+2,3); BC!(c,U)
    a = Flow(u,c,U,Δt=0.1,ν=0.01)
    b = MultiLevelPS(c)
    for n ∈ N
        @show n
        for i ∈ 1:n
            mom_step!(a,b)
        end
        @btime mom_step!($a,$b)
    end
end

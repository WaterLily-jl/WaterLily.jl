using WaterLily
using Plots

function TwoD_block_video(p,Re=250)
    n,m = 2^p,2^(p-1); xr = m÷2:m÷2+1; yr = 3m÷8+2:5m÷8+1;
    U,L = [1.,0.],m/4.; ν=U[1]*L/Re
    @show L,ν

    u = zeros(n+2,m+2,2); BC!(u,U)
    c = ones(n+2,m+2,2); BC!(c,[0. 0.])

    # immerse a solid block (proto-BDIM)
    u[first(xr):last(xr)+1,yr,1] .= 0
    c[first(xr):last(xr)+1,yr,1] .= 0
    c[xr,first(yr):last(yr)+1,2] .= 0

    a = Flow(u,c,U,ν=ν);
    b = MultiLevelPS(a.μ₀)
    ω₃ = zeros(n,m)

    tprint,Δprint,nprint = -30.,0.25,100
    gr(show = false, size=(780,360))
    @time @gif for i ∈ 1:nprint
        while tprint<0
            mom_step!(a,b)
            tprint += a.Δt[end]*U[1]/L
        end
        @inside ω₃[I]=WaterLily.curl(3,I,a.u)*L/U[1]
        flood(collect(1:n).-0.5,collect(1:m).-0.5,ω₃,clims=(-16,16))
        body([xr[1],xr[end],xr[end],xr[1],xr[1]],[yr[1],yr[1],yr[end],yr[end],yr[1]])
        tprint-=Δprint
    end
    return a
end

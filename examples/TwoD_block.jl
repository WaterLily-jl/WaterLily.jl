using WaterLily
using Plots

function TwoD_block_video(p,Re=250)
    # Set simulation size & physical parameters
    n,m = 2^p,2^(p-1)
    U,L = [1.,0.],m/4.; ν=U[1]*L/Re
    @show L,ν

    # Initialize uniform fields
    u = zeros(n+2,m+2,2); BC!(u,U)
    c = ones(n+2,m+2,2); BC!(c,[0. 0.])
    ω₃ = zeros(n,m)

    # Immerse a solid block over an x,y range
    xr,yr = m÷2:m÷2+1,3m÷8+2:5m÷8+1;
    u[xr[1]:xr[end]+1,yr,1] .= 0
    c[xr[1]:xr[end]+1,yr,1] .= 0
    c[xr,yr[1]:yr[end]+1,2] .= 0
    x = [xr[1],xr[end],xr[end],xr[1],xr[1]]
    y = [yr[1],yr[1],yr[end],yr[end],yr[1]]

    # Initialize flow and Poisson system
    a = Flow(u,c,U,ν=ν);
    b = MultiLevelPS(a.μ₀)

    # Evolve solution in time and plot to a gif
    tprint,Δprint,nprint = -30.,0.25,100
    gr(show = false, size=(780,360))
    @time @gif for i ∈ 1:nprint
        while tprint<0
            mom_step!(a,b)
            tprint += a.Δt[end]*U[1]/L
        end
        @inside ω₃[I]=WaterLily.curl(3,I,a.u)*L/U[1]
        flood(ω₃,shift=(-0.5,-0.5),clims=(-16,16))
        addbody(x,y)
        tprint-=Δprint
    end
    return a
end

using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function TwoD_circle_video(p,Re=250)
    # Set simulation size & physical parameters
    n,m = 2^p,2^(p-1)
    U,R,center = [1.,0.], m/8., [m/2,m/2]
    ν=U[1]*R/Re
    @show R,ν

    # Immerse a circle (change for other shapes)
    c = BDIM_coef(n+2,m+2,2) do xy
        norm2(xy .- center) - R  # signed distance function
    end

    # Initialize flow and Poisson system
    u = zeros(n+2,m+2,2); BC!(u,U)
    a = Flow(u,c,U,ν=ν);
    b = MultiLevelPoisson(c)

    # Evolve solution in time and plot to a gif
    tprint,Δprint,nprint = 0.0,0.25,200
    gr(show = false, size=(780,360))
    @time @gif for i ∈ 1:nprint
        while tprint<0
            mom_step!(a,b)
            tprint += a.Δt[end]*U[1]/R
        end
        @inside a.σ[I]=WaterLily.curl(3,I,a.u)*R/U[1]
        flood(a.σ,shift=(-0.5,-0.5),clims=(-5,5))
        tprint-=Δprint
    end
    return a
end

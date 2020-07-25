using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function TwoD_julia_video(p=6,Re=250)
    # Set simulation size & physical parameters
    n,m = 2^p,2^p
    U,R,r = [0.,-1.], m/16, m/16/0.75
    ν=norm2(U)*R/Re
    @show R,ν

    # Immerse three well placed circles (change for other shapes)
    centers = [n/2+im*3*m/4+r*exp(im*ϕ) for ϕ ∈ [π/2,π/2+2π/3,π/2-2π/3]]
    colors = [:forestgreen,:brown3,:mediumorchid3]

    c = BDIM_coef(n+2,m+2,2) do xy  # signed distance function
            minimum(centers) do center
                norm2(complex(xy...) - center) - R
            end
    end

    # Initialize flow and Poisson system
    u = zeros(n+2,m+2,2); BC!(u,U)
    a = Flow(u,c,U,ν=ν);
    b = MultiLevelPoisson(c)

    # Evolve solution in time and plot to a gif
    tprint,Δprint,nprint = 0.0,0.3,200
    gr(show = false, size=(700,600))
    @time @gif for i ∈ 1:nprint
        while tprint<0
            mom_step!(a,b)
            tprint += a.Δt[end]*norm2(U)/R
        end
        @inside a.σ[I]=WaterLily.curl(3,I,a.u)*R/norm2(U)
        flood(a.σ,shift=(-0.5,-0.5),clims=(-5,5),
            cfill=:Blues,legend=false,border=:none)
        for (center,color) ∈ zip(centers,colors)
            z = [R*exp(im*θ)+center for θ ∈ range(0,2π,length=33)]
            addbody(real(z),imag(z),c=color)
        end
        tprint-=Δprint
    end
    return a
end

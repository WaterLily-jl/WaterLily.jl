using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function TwoD_julia_video(;p=6,Re=250,stop=60.)
    # Set simulation size & physical parameters
    n,m = 2^p,2^p
    U,R,r = [0.,-1.], m/16, m/16/0.75
    ν=norm2(U)*R/Re
    @show R,ν

    # Immerse three well placed circles (change for other shapes)
    z = [R*exp(im*θ) for θ ∈ range(0,2π,length=33)]
    centers = [n/2+im*3*m/4+r*exp(im*ϕ) for ϕ ∈ [π/2,π/2+2π/3,π/2-2π/3]]
    colors = [:forestgreen,:brown3,:mediumorchid3]
    body = AutoBody() do x,t  # signed distance function
            minimum(centers) do center
                norm2(complex(x...) - center) - R
            end
    end

    # Initialize simulation
    sim = Simulation((n+2,m+2),U,R;ν,body)

    # Solve flow and make nice gif
    t = range(0.,stop;step=0.3)
    gr(show = false, size=(700,600))
    @time @gif for tᵢ in t
        println("tU/L=",round(tᵢ,digits=4))
        sim_step!(sim,tᵢ)
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ,shift=(-0.5,-0.5),clims=(-5,5),
            cfill=:Blues,legend=false,border=:none)
        for (center,color) ∈ zip(centers,colors)
            addbody(real(z.+center),imag(z.+center),c=color)
        end
    end
    return sim
end

using WaterLily
using StaticArrays
using CUDA

include("TwoD_plots.jl")

function run_TwoD_julia(;p=6,Re=400,stop=100.)
    # Set simulation size & physical parameters
    n,m = 2^p,2^p
    U,R,r = 1, m/16, m/16/0.75

    # Immerse three well placed circles (change for other shapes)
    z = @SArray [R*exp(im*θ) for θ ∈ range(0,2π,length=33)]
    centers = @SArray [2n/2+im*14*m/4+r*exp(im*ϕ) for ϕ ∈ [π/2,π/2+2π/3,π/2-2π/3]]
    colors = [:forestgreen,:brown3,:mediumorchid3]
    body = AutoBody() do x,t  # signed distance function
            minimum(centers) do center
                √sum(abs2,complex(x...) - center) - R
            end
    end

    # Initialize simulation on the GPU and a CPU buffer array for plotting
    sim = Simulation((2n,4m),(0,-U),R;ν=U*R/Re,body,mem=CuArray)
    σ = zeros(size(sim.flow.σ))

    # Solve flow and make nice gif
    @time @gif for tᵢ in range(0.,stop;step=0.5)
        println("tU/L=",round(tᵢ,digits=4))
        sim_step!(sim,tᵢ)
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*R/U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I])<0.001,0.0,sim.flow.σ[I])
        copyto!(σ,sim.flow.σ)
        flood(σ,shift=(-2,-1.5),clims=(-5,5),
            cfill=:seismic,legend=false,border=:none,size=(2n,4m))
        for (center,color) ∈ zip(centers,colors)
            addbody(real(z.+center),imag(z.+center),c=color)
        end
    end
end

run_TwoD_julia(;p=8)
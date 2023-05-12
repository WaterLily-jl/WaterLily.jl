using WaterLily
function TGV(; pow=6, Re=1e5, T=Float32, mem=Array)
    # Define vortex size, velocity, viscosity
    L = 2^pow; U = 1; ν = U*L/Re
    # Taylor-Green-Vortex initial velocity field
    function uλ(i,xyz)
        x,y,z = @. (xyz-1.5)*π/L                # scaled coordinates
        i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x
        i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y
        return 0.                              # u_z
    end
    # Initialize simulation
    return Simulation((L, L, L), (0, 0, 0), L; U, uλ, ν, T, mem)
end

# Initialize CUDA simulation
using CUDA: CUDA
@assert CUDA.functional()
sim = TGV(mem=CUDA.CuArray);

# Create a video using Makie
dat = sim.flow.σ[inside(sim.flow.σ)] |> Array; # CPU buffer array
function λ₂!(dat,sim)                          # compute log10(-λ₂)
    a = sim.flow.σ
    @inside a[I] = log10(max(1e-6,-WaterLily.λ₂(I,sim.flow.u)*sim.L/sim.U))
    copyto!(dat,a[inside(a)])                  # copy to CPU
end
include("ThreeD_Plots.jl")
@time makie_video!(sim,dat,λ₂!,name="TGV.mp4",duration=5) do obs
    contour(obs,levels=[-3,-2,-1,0],alpha=0.1,isorange=0.5)
end

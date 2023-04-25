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

using CUDA: CUDA
@assert CUDA.functional()
sim = TGV(mem=CUDA.CuArray);

include("ThreeD_Plots.jl")
dat = sim.flow.σ[inside(sim.flow.σ)] |> Array; # CPU buffer array
function ω_mag!(dat,sim)      # compute |ω|
    a = sim.flow.σ
    @inside a[I] = WaterLily.ω_mag(I,sim.flow.u)*sim.L/sim.U
    copyto!(dat,a[inside(a)]) # copy to CPU
end
@time volume_video!(sim,dat,ω_mag!,name="TGV.mp4",duration=5);

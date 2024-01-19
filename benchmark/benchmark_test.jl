using WaterLily
using BenchmarkTools
using CUDA: CuArray, allowscalar
using KernelAbstractions: synchronize, get_backend
using StaticArrays

allowscalar(false)
include("util.jl")

cases, log2p, max_steps, ftype, backend = ["tgv"], [(6,)], [1], [Float32], Array

# Define simulation benchmarks
function tgv(p, backend; Re=1e5, T=Float32)
    # Define vortex size, velocity, viscosity
    L = 2^p; U = 1; ν = U*L/Re
    # Taylor-Green-Vortex initial velocity field
    function uλ(i,vx)
        x,y,z = @. (vx-1.5)*π/L                # scaled coordinates
        i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x
        i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y
        return 0.                              # u_z
    end
    # Initialize simulation
    return Simulation((L,L,L), (0,0,0), L; U=U, uλ=uλ, ν=ν, T=T, mem=backend)
end

# Generate benchmarks
# function benchmark()
#     for (case, p, s, ft) in zip(cases, log2p, max_steps, ftype)
#         println("Benchmarking: $(case)")
#         s = getf(case)(p[1], backend; T=ft)
#         # sim_step!(s, typemax(ft); max_steps=2, verbose=false, remeasure=false)
#         # @time sim_step!(s, typemax(ft); max_steps=10, verbose=false, remeasure=false)
#         @btime sim_step!($s, typemax($ft); max_steps=10, verbose=false, remeasure=false) samples=1 evals=1
#     end
# end

# benchmark()
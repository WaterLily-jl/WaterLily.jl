using WaterLily
using BenchmarkTools
using JLD2
using LinearAlgebra: norm2

function TGV(p; Re=1e5, T=Float32)
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
    return Simulation((L+2,L+2,L+2),zeros(3),L;U,uλ,ν,T)
end

function donut(p; Re=1e3, T=Float32)
    # Define simulation size, geometry dimensions, viscosity
    n = 2^p
    center,R,r = [n/2,n/2,n/2], n/4, n/16
    ν = R/Re
    # Apply signed distance function for a torus
    body = AutoBody() do xyz,t
        x,y,z = xyz - center
        norm2([x,norm2([y,z])-R])-r
    end
    return Simulation((2n+2,n+2,n+2),[1.,0.,0.],R;ν,body,T)
end

log2N, t_sim_CTU, T = (4, 5, 6, 7), 0.1, Float32

# Force first run to compile
sim_temp = donut(4; T=T)
sim_step!(sim_temp, t_sim_CTU; verbose=true, remeasure=false)

# Create benchmark suite
suite = BenchmarkGroup()
for n ∈ log2N
    suite[repr(n)] = BenchmarkGroup([repr(n)])
    sim = donut(n; T=T)
    suite[repr(n)]["sim_step!"] = @benchmarkable sim_step!($sim, $t_sim_CTU; verbose=true, remeasure=false)
end

# Run benchmarks
samples = 1 # We can only use 1 sample since more than once used that last flow.time and does not iterate further.
evals = 1 # better to use evaulations instead
verbose = true
r = run(suite, samples = samples, evals = evals, seconds = 1e6, verbose = verbose)
save_object("benchmark/donut/sim_step_master_3D_5678.dat", r)
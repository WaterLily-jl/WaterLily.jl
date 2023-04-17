using WaterLily
using BenchmarkTools
using JLD2

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

log2N, t_sim_CTU, T = (5, 6, 7, 8), 0.1, Float32
# log2N, t_sim_CTU, T = (5,), 0.1, Float32

# Force first run to compile
sim_temp = TGV(5; T=T)
sim_step!(sim_temp, t_sim_CTU; verbose=true, remeasure=false)

suite = BenchmarkGroup()
for n ∈ log2N
    suite[repr(n)] = BenchmarkGroup([repr(n)])
    sim = TGV(n; T=T)
    suite[repr(n)]["sim_step!"] = @benchmarkable sim_step!($sim, $t_sim_CTU; verbose=true, remeasure=false)
end

# Run benchmarks
samples = 1 # We can only use 1 sample since more than once used that last flow.time and does not iterate further.
evals = 1 # better to use evaulations instead
verbose = true
r = run(suite, samples = samples, evals = evals, seconds = 1e6, verbose = verbose)
save_object("benchmark/tgv/sim_step_master_3D_5678.dat", r)
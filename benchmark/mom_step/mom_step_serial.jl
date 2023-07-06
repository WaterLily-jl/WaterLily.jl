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
    return Flow((L+2,L+2,L+2),zeros(3); ν, uλ, T)
end

log2N = (5, 6, 7, 8)
T = Float32
U = T[0.0, 0.0, 0.0]

suite = BenchmarkGroup()
for n ∈ log2N
    flow = TGV(n; T=T)
    pois = MultiLevelPoisson(flow.μ₀)
    suite[repr(n)] = BenchmarkGroup([repr(n)])
    suite[repr(n)]["conv_diff!"] = @benchmarkable WaterLily.conv_diff!($flow.f, $flow.u⁰, ν=$flow.ν)
    suite[repr(n)]["BDIM!"] = @benchmarkable WaterLily.BDIM!($flow)
    suite[repr(n)]["BC!"] = @benchmarkable BC!($flow.u, $flow.U)
    suite[repr(n)]["project!"] = @benchmarkable WaterLily.project!($flow, $pois)
    suite[repr(n)]["CFL"] = @benchmarkable WaterLily.CFL($flow)
end

# Run benchmarks
samples = 100 # Use >1 since timings reported are min(samples), and the first run always compiles
verbose = true
r = run(suite, samples = samples, seconds = 1e6, verbose = verbose)
save_object("benchmark/mom_step/mom_step_5678_serial.dat", r)
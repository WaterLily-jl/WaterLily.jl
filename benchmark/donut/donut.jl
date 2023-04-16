using WaterLily
using LinearAlgebra: norm2
using BenchmarkTools
using CUDA: CuArray
using KernelAbstractions: synchronize, get_backend
using JLD2

macro add_benchmark(args...)
    ex, b, suite, label = args
    return quote
        $suite[$label] = @benchmarkable begin
            $ex
            synchronize($b)
        end
    end |> esc
end

# function TGV(p, backend; Re=1e5, T=Float32)
#     # Define vortex size, velocity, viscosity
#     L = 2^p; U = 1; ν = U*L/Re
#     # Taylor-Green-Vortex initial velocity field
#     function uλ(i,vx)
#         x,y,z = @. (vx-1.5)*π/L                # scaled coordinates
#         i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x
#         i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y
#         return 0.                              # u_z
#     end
#     # Initialize simulation
#     return Simulation((L, L, L), (0, 0, 0), L; U=U, uλ=uλ, ν=ν, T=T, mem=backend)
# end

function donut(p, backend; Re=1e3, T=Float32)
    # Define simulation size, geometry dimensions, viscosity
    n = 2^p
    center,R,r = (n/2,n/2,n/2), n/4, n/16
    ν = R/Re
    # Apply signed distance function for a torus
    # sdf(x,t) = norm2((x[1]-center[1], norm2((x[2]-center[2],x[3]-center[3]))-R))-r
    # sdf(x,t) = √sum(abs2,(x[1], √sum(abs2, (x[2], x[3])) - R))
    # body = AutoBody(sdf)
    body = AutoBody() do xyz, t
        x,y,z = xyz .- center
        √sum(abs2,(x, √sum(abs2, (y, z)) - R)) # norm2([x,norm2([y,z])-R])-r
    end
    # Initialize simulation
    Simulation((2n, n, n), (1, 0, 0), R; ν=ν, body=body, T=T, mem=backend)
end

function create_suite()
    suite = BenchmarkGroup()
    for (ArrayT, b) ∈ zip([Array, CuArray], backends_str)
        suite[b] = BenchmarkGroup([b])
        for n ∈ log2N
            sim = donut(n, ArrayT; T=T)
            backend = get_backend(sim.flow.p)
            suite[b][repr(n)] = BenchmarkGroup([repr(n)])
            @add_benchmark sim_step!($sim, $t_sim_CTU; verbose=true, remeasure=false) $backend suite[b][repr(n)] "sim_step!"
        end
    end
    return suite
end

log2N, t_sim_CTU, T = (4, 5, 6, 7), 0.1, Float32

backends_str = ["CPU", "GPU"]
r = BenchmarkGroup()
samples = 1 # We can't only use >1 samples since flow reaches flow.time on the first one and does not iterate further.
evals = 1
verbose = true
save_benchmark = false
run_benchmarks = false

# Run or load benchmarks
if run_benchmarks
    # Force first run to compile
    simCPU = donut(3, Array; T=T)
    sim_step!(simCPU, t_sim_CTU; verbose=true, remeasure=false)
    simGPU = donut(3, CuArray; T=T)
    sim_step!(simGPU, t_sim_CTU; verbose=true, remeasure=false)
    # Create benchmark suite
    suite = create_suite()
    r["CPU"] = run(suite["CPU"], samples = samples, evals = evals, seconds = 1e6, verbose = verbose)
    r["GPU"] = run(suite["GPU"], samples = samples, evals = evals,  seconds = 1e6, verbose = verbose)
    save_benchmark && save_object("benchmark/donut/sim_step_CUDA_3D_5678.dat", r)
else
    r = load_object("benchmark/donut/sim_step_CUDA_3D_5678.dat")
end
# Serial (master) benchmarks
r["serial"] = load_object("benchmark/donut/sim_step_master_3D_5678.dat")

# Postprocess results
push!(backends_str, "serial")
btimes = Dict((b, Dict((n, 0.0) for n ∈ repr.(log2N))) for b ∈ backends_str)
for b ∈ backends_str, n ∈ repr.(log2N)
    btimes[b][n]= r[b][n]["sim_step!"] |> time # only single sample
    btimes[b][n]/= 10^6 # times now in ms
end
btimes_sim_step = (serial = T[btimes["serial"][n] for n ∈ repr.(log2N)],
                      CPU = T[btimes["CPU"][n] for n ∈ repr.(log2N)],
                      GPU = T[btimes["GPU"][n] for n ∈ repr.(log2N)])

# speedups
using Printf
println("\nSpeedups: n | routine | CPU | GPU")
for n ∈ repr.(log2N)
    @printf("\nn=%s | %10s |  %4.2f | %4.2f",
        n, "sim_step!", btimes["serial"][n]/btimes["CPU"][n], btimes["serial"][n]/btimes["GPU"][n])
end

# Plots
# using Plots, LaTeXStrings
# p1 = plot(size=(600,600), xlabel=L"\log_2(N)", ylabel="Donut sim_step! "* L"[ms]",
#     yscale=:log10, legend=:bottomright, foreground_color_legend=nothing,
#     yticks=[10.0^n for n in 2:5])
# plot!([n for n ∈ repr.(log2N.*3)], btimes_sim_step[:serial], label="serial", marker=4, color=:red)
# plot!([n for n ∈ repr.(log2N.*3)], btimes_sim_step[:CPU], label="CPU", marker=4, color=:blue)
# plot!([n for n ∈ repr.(log2N.*3)], btimes_sim_step[:GPU], label="GPU", marker=4, color=:green)

# Plots.scalefontsizes(1.5)
# savefig("benchmark/donut/donut_benchmark.pdf");
# Plots.scalefontsizes()
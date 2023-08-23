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

function TGV(p, backend; Re=1e5, T=Float32)
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
    return Simulation((L, L, L), (0, 0, 0), L; U=U, uλ=uλ, ν=ν, T=T, mem=backend)
end

function create_suite()
    suite = BenchmarkGroup()
    for (ArrayT, b) ∈ zip([Array, CuArray], backends_str)
        suite[b] = BenchmarkGroup([b])
        for n ∈ log2N
            sim = TGV(n, ArrayT; T=T)
            backend = get_backend(sim.flow.p)
            suite[b][repr(n)] = BenchmarkGroup([repr(n)])
            @add_benchmark sim_step!($sim, $t_sim_CTU; verbose=true, remeasure=false) $backend suite[b][repr(n)] "sim_step!"
        end
    end
    return suite
end

log2N, t_sim_CTU, T = (5, 6, 7, 8), 0.1, Float32

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
    simCPU = TGV(4, Array; T=T)
    sim_step!(simCPU, t_sim_CTU; verbose=true, remeasure=false)
    simGPU = TGV(4, CuArray; T=T)
    sim_step!(simGPU, t_sim_CTU; verbose=true, remeasure=false)
    # Create benchmark suite
    suite = create_suite()
    r["CPU"] = run(suite["CPU"], samples = samples, evals = evals, seconds = 1e6, verbose = verbose)
    r["GPU"] = run(suite["GPU"], samples = samples, evals = evals,  seconds = 1e6, verbose = verbose)
    # save_benchmark && save_object("benchmark/tgv/sim_step_5678_update_mult_1.9.2.dat", r)
    save_benchmark && save_object("benchmark/tgv/sim_step_5678_master_1.9.2.dat", r)
else
    # r = load_object("benchmark/tgv/sim_step_5678_update_mult_1.9.2.dat")
    r = load_object("benchmark/tgv/sim_step_5678_master_1.9.2.dat")
end
# Serial (master) benchmarks
# r["serial"] = load_object("benchmark/tgv/sim_step_5678_serial_1.8_old.dat")
r["serial"] = load_object("benchmark/tgv/sim_step_5678_serial_1.8.5.dat")

# Postprocess results
push!(backends_str, "serial")
btimes = Dict((b, Dict((n, 0.0) for n ∈ repr.(log2N))) for b ∈ backends_str)
for b ∈ backends_str, n ∈ repr.(log2N)
    btimes[b][n]= r[b][n]["sim_step!"] |> time # only single sample
    btimes[b][n]/= 10^9 # times now in ms
end
btimes_sim_step = (serial = T[btimes["serial"][n] for n ∈ repr.(log2N)],
                      CPU = T[btimes["CPU"][n] for n ∈ repr.(log2N)],
                      GPU = T[btimes["GPU"][n] for n ∈ repr.(log2N)])

# speedups
using Printf
println("\nSpeedups:\n n  |   routine  |  CPU   |  GPU\n----------------------------------")
for n ∈ repr.(log2N)
    @printf("n=%s | %10s | %06.2f | %06.2f\n",
        n, "sim_step!", btimes["serial"][n]/btimes["CPU"][n], btimes["serial"][n]/btimes["GPU"][n])
end

# Plots
# using Plots, LaTeXStrings
# p1 = plot(size=(600,600), xlabel=L"\log_2(N)", ylabel="TGV sim_step! "* L"[s]",
#     yscale=:log10, legend=:bottomright, foreground_color_legend=nothing, legendfontsize=12,
#     yticks=[10.0^n for n in -1:2], markerstrokewidth=0)
# plot!([n for n ∈ repr.(log2N.*3)], btimes_sim_step[:serial], label="serial", marker=4, color=:red, markerstrokewidth=0.25)
# plot!([n for n ∈ repr.(log2N.*3)], btimes_sim_step[:CPU], label="CPU", marker=4, color=:blue, markerstrokewidth=0.25)
# plot!([n for n ∈ repr.(log2N.*3)], btimes_sim_step[:GPU], label="GPU", marker=4, color=:green, markerstrokewidth=0.25)

# Plots.scalefontsizes(1.5)
# savefig("benchmark/tgv/tgv_benchmark.pdf");
# Plots.scalefontsizes()
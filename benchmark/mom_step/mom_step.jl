using WaterLily
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

function create_suite()
    suite = BenchmarkGroup()
    for (ArrayT, b) ∈ zip([Array, CuArray], backends_str)
        suite[b] = BenchmarkGroup([b])
        for n ∈ log2N
            flow = TGV(n, ArrayT;  T=T)
            pois = MultiLevelPoisson(flow.p, flow.μ₀, flow.σ)
            backend = get_backend(flow.p)
            suite[b][repr(n)] = BenchmarkGroup([repr(n)])
            @add_benchmark WaterLily.conv_diff!($flow.f, $flow.u⁰, $flow.σ, ν=$flow.ν) $backend suite[b][repr(n)] "conv_diff!"
            @add_benchmark WaterLily.BDIM!($flow) $backend suite[b][repr(n)] "BDIM!"
            @add_benchmark BC!($flow.u, $flow.U) $backend suite[b][repr(n)] "BC!"
            @add_benchmark WaterLily.project!($flow, $pois) $backend suite[b][repr(n)] "project!"
            @add_benchmark WaterLily.CFL($flow) $backend suite[b][repr(n)] "CFL"
        end
    end
    return suite
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
    return Flow((L, L, L), (0, 0, 0); f=backend, ν=ν, uλ=uλ, T=T)
end

log2N = (5, 6, 7, 8)
U, T = (0, 0, 0), Float32

backends_str = ["CPU", "GPU"]
r = BenchmarkGroup()
samples = 100 # Use >1 since timings reported are min(samples), and the first run always compiles
verbose = true
save_benchmark = false
run_benchmarks = false

# Run or load benchmarks
if run_benchmarks
    suite = create_suite()
    r["CPU"] = run(suite["CPU"], samples = samples, seconds = 1e6, verbose = verbose)
    r["GPU"] = run(suite["GPU"], samples = samples, seconds = 1e6, verbose = verbose)
    save_benchmark && save_object("benchmark/mom_step/mom_step_CUDA_3D_5678.dat", r)
else
    r = load_object("benchmark/mom_step/mom_step_CUDA_3D_5678.dat")
end
# Serial (master) benchmarks
r["serial"] = load_object("benchmark/mom_step/mom_step_master_3D_5678.dat")

# Postprocess results
routines = ["conv_diff!", "BDIM!", "BC!", "project!", "CFL"]
push!(backends_str, "serial")
btimes = Dict((b, Dict((n, Dict()) for n ∈ repr.(log2N))) for b ∈ backends_str)
for b ∈ backends_str, n ∈ repr.(log2N), f ∈ routines
    btimes[b][n][f] = r[b][n][f][2:end] |> minimum |> time # throw out first sample
    btimes[b][n][f] /= 10^6 # times now in ms
end
btimes_conv_diff = (serial = T[btimes["serial"][n]["conv_diff!"] for n ∈ repr.(log2N)],
                      CPU = T[btimes["CPU"][n]["conv_diff!"] for n ∈ repr.(log2N)],
                      GPU = T[btimes["GPU"][n]["conv_diff!"] for n ∈ repr.(log2N)])
btimes_project = (serial = T[btimes["serial"][n]["project!"] for n ∈ repr.(log2N)],
                    CPU = T[btimes["CPU"][n]["project!"] for n ∈ repr.(log2N)],
                    GPU = T[btimes["GPU"][n]["project!"] for n ∈ repr.(log2N)])

# speedups
using Printf
println("\nSpeedups:\n n  |   routine  |  CPU   |  GPU\n----------------------------------")
for n ∈ repr.(log2N), f ∈ routines
    @printf("n=%s | %10s | %06.2f | %06.2f\n",
        n, f, btimes["serial"][n][f]/btimes["CPU"][n][f], btimes["serial"][n][f]/btimes["GPU"][n][f])
end

# Plots
# using Plots, LaTeXStrings
# p1 = plot(size=(600,600), xlabel=L"\log_2(N)", ylabel="TGV conv_diff! "* L"[ms]",
#     yscale=:log10, legend=:bottomright, foreground_color_legend=nothing, legendfontsize=12,
#     yticks=[10.0^n for n in 0:3])
# plot!([n for n ∈ repr.(log2N.*3)], btimes_conv_diff[:serial], label="serial", marker=4, color=:red, markerstrokewidth=0.25)
# plot!([n for n ∈ repr.(log2N.*3)], btimes_conv_diff[:CPU], label="CPU", marker=4, color=:blue, markerstrokewidth=0.25)
# plot!([n for n ∈ repr.(log2N.*3)], btimes_conv_diff[:GPU], label="GPU", marker=4, color=:green, markerstrokewidth=0.25)
# Plots.scalefontsizes(1.5)
# savefig("benchmark/mom_step/benchmark_tgv_conv_diff.pdf");
# Plots.scalefontsizes()

# p1 = plot(size=(600,600), xlabel=L"\log_2(N)", ylabel="TGV project! "* L"[ms]",
#     yscale=:log10, legend=:bottomright, foreground_color_legend=nothing, legendfontsize=12,
#     yticks=[10.0^n for n in 0:3])
# plot!([n for n ∈ repr.(log2N.*3)], btimes_project[:serial], label="serial", marker=4, color=:red, markerstrokewidth=0.25)
# plot!([n for n ∈ repr.(log2N.*3)], btimes_project[:CPU], label="CPU", marker=4, color=:blue, markerstrokewidth=0.25)
# plot!([n for n ∈ repr.(log2N.*3)], btimes_project[:GPU], label="GPU", marker=4, color=:green, markerstrokewidth=0.25)
# Plots.scalefontsizes(1.5)
# savefig("benchmark/mom_step/benchmark_tgv_project.pdf");
# Plots.scalefontsizes()


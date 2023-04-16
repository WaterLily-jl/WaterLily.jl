using WaterLily
using BenchmarkTools
using JLD2

log2N = (5, 6, 7, 8)
T = Float32
U = T[0.0, 0.0, 0.0]

suite = BenchmarkGroup()
for n ∈ log2N
    flow = Flow((2^n+2, 2^n+2, 2^n+2), U; T=T)
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
save_object("benchmark/mom_step/mom_step_master_3D_5678.dat", r)

# Postprocess results
# minimum timings (in ns)
# routines = ["conv_diff!", "BDIM!", "BC!", "project!", "CFL"]
# time_min = Dict((n, Dict()) for n ∈ repr.(log2N))
# for n ∈ repr.(log2N), f ∈ routines
#     time_min[n][f] = r[n][f] |> minimum |> time
#     time_min[n][f] /= 10^6
# end
# time_min_conv_diff = T[time_min[n]["conv_diff!"] for n ∈ repr.(log2N)]
# time_min_project = T[time_min[n]["project!"] for n ∈ repr.(log2N)]

# Plots
# using Plots, LaTeXStrings
# p = plot(xlabel=L"\log_2(N)", ylabel="conv_diff! " * L"[{\mu}s]", yscale=:log10)
# plot!([n for n ∈ repr.(log2N)], time_min_conv_diff, label="CPU", marker=4, color=:blue)
using WaterLily
using BenchmarkTools
using CUDA: CuArray
using KernelAbstractions: synchronize, get_backend

macro add_benchmark(args...)
    ex, b, suite, label = args
    return quote
        $suite[$label] = @benchmarkable begin
            $ex
            synchronize($b)
        end
    end |> esc
end

log2N = (6, 7, 8, 9, 10, 11, 12)
U, T = (0, 0), Float32

suite = BenchmarkGroup()
arrayTs, backends_str = [Array, CuArray], ["CPU", "GPU"]
backends = [get_backend(rand(1) |> arrayT) for arrayT ∈ arrayTs]

for (ArrayT, b) ∈ zip([Array, CuArray], backends_str)
    suite[b] = BenchmarkGroup([b])
    for n ∈ log2N
        flow = Flow((2^n, 2^n), U; f=ArrayT, T=T)
        pois = MultiLevelPoisson(flow.p, flow.μ₀, flow.σ)
        backend = get_backend(flow.p)
        suite[b][repr(n)] = BenchmarkGroup([repr(n)])
        @add_benchmark WaterLily.conv_diff!($flow.f, $flow.u⁰, $flow.σ, ν=$flow.ν) backend suite[b][repr(n)] "conv_diff!"
        @add_benchmark WaterLily.BDIM!($flow) backend suite[b][repr(n)] "BDIM!"
        @add_benchmark BC!($flow.u, $flow.U) backend suite[b][repr(n)] "BC!"
        @add_benchmark WaterLily.project!($flow, $pois) backend suite[b][repr(n)] "project!"
        @add_benchmark WaterLily.CFL($flow) backend suite[b][repr(n)] "CFL"
    end
end

# Run benchmarks
r = BenchmarkGroup()
samples = 1000 # Use >1 since timings reported are min(samples), and the first run always compiles
verbose = true
# CPU run benchmarks
backend = backends[1]
r["CPU"] = run(suite["CPU"], samples = samples, verbose = verbose)
# GPU run benchmarks
backend = backends[2]
r["GPU"] = run(suite["GPU"], samples = samples, verbose = verbose)

# Postprocess results
# minimum timings (in ns)
routines = ["conv_diff!", "BDIM!", "BC!", "project!", "CFL"]
time_min = Dict((b, Dict((n, Dict()) for n ∈ repr.(log2N))) for b ∈ backends_str)
for b ∈ backends_str, n ∈ repr.(log2N), f ∈ routines
    time_min[b][n][f] = r[b][n][f] |> minimum |> time
end
time_min_conv_diff = (CPU = T[time_min["CPU"][n]["conv_diff!"] for n ∈ repr.(log2N)]./10^3,
                      GPU = T[time_min["GPU"][n]["conv_diff!"] for n ∈ repr.(log2N)]./10^3)
time_min_project = (CPU = T[time_min["CPU"][n]["project!"] for n ∈ repr.(log2N)]./10^3,
                    GPU = T[time_min["GPU"][n]["project!"] for n ∈ repr.(log2N)]./10^3)

# Plots
using Plots, LaTeXStrings
p = plot(xlabel=L"\log_2(N)", ylabel="conv_diff! " * L"[{\mu}s]", yscale=:log10)
plot!([n for n ∈ repr.(log2N)], time_min_conv_diff[:CPU], label="CPU", marker=4, color=:blue)
plot!([n for n ∈ repr.(log2N)], time_min_conv_diff[:GPU], label="GPU", marker=4, color=:green)
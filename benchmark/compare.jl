using BenchmarkTools, PrettyTables

# Load benchmarks
benchmarks = [BenchmarkTools.load(f)[1] for f in ARGS]
# Get backends string vector and assert same case sizes for the different backends
backends_str = [String.(k)[1] for k in keys.(benchmarks)]
log2p_str = [String.(keys(benchmarks[i][backend_str])) for (i, backend_str) in enumerate(backends_str)]
@assert length(unique(log2p_str)) == 1
# Assuming the case and tested function is the same in all benchmarks, we grab their name
case, f_test = benchmarks[1].tags[1:2]
# Get data for PrettyTables
header = ["Backend", "WaterLily", "Julia", "Precision", "Allocations", "GC [%]", "Time [s]", "Speed-up"]
data, base_speedup = Matrix{Any}(undef, length(benchmarks), length(header)), 1.0
printstyled("Benchmark environment: $case $f_test (max_steps=$(benchmarks[1].tags[4]))\n", bold=true)
for n in log2p_str[1]
    printstyled("▶ log2p = $n\n", bold=true)
    for (i, benchmark) in enumerate(benchmarks)
        datap = benchmark[backends_str[i]][n][f_test]
        speedup = i == 1 ? 1.0 : benchmarks[1][backends_str[1]][n][f_test].times[1] / datap.times[1]
        data[i, :] .= [backends_str[i], benchmark.tags[end-1], benchmark.tags[end], benchmark.tags[end-3],
            datap.allocs, (datap.gctimes[1] / datap.times[1]) * 100.0, datap.times[1] / 1e9, speedup]
    end
    pretty_table(data; header=header, header_alignment=:c, formatters=ft_printf("%.2f", [6,7,8]))
end


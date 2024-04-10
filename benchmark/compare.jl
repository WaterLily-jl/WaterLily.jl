using BenchmarkTools, PrettyTables

# Load benchmarks
benchmarks = [BenchmarkTools.load(fname)[1] for fname in ARGS if !occursin("--sort", fname)]
sort_cla = findfirst(occursin.("--sort", ARGS))
sort_idx = !isnothing(sort_cla) ? ARGS[sort_cla] |> x -> split(x, "=")[end] |> x -> parse(Int, x) : 0
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
    sorted_cond, sorted_idx = 0 < sort_idx <= 8, nothing
    if sorted_cond
        sorted_idx = sortperm(data[:, sort_idx])
        baseline_idx = findfirst(x->x==1, sorted_idx)
        data .= data[sorted_idx, :]
    end
    hl_base = Highlighter(f=(data, i, j) -> sorted_cond ? i == findfirst(x->x==1, sorted_idx) : i==1,
        crayon=Crayon(foreground=:blue))
    hl_fast = Highlighter(f=(data, i, j) -> i == argmin(data[:, end-1]), crayon=Crayon(foreground=(32,125,56)))
    pretty_table(data; header=header, header_alignment=:c, highlighters=(hl_base, hl_fast), formatters=ft_printf("%.2f", [6,7,8]))
end


# Automatic benchmark generation suite

Suite to generate benchmarks across different Julia versions (using [juliaup](https://github.com/JuliaLang/juliaup)), backends, cases, and cases sizes using the [benchmark.sh](./benchmark.sh) script.

## TL;DR
Usage example
```
sh benchmark.sh  -v "1.9.4 1.10.0-rc1" -t "1 3 6" -b "Array CuArray" -c "tgv.jl" -p "5,6,7"
```
The default launch is equivalent to:
```
sh benchmark.sh  -v JULIA_USER_VERSION -t "1 6" -b "Array CuArray" -c "tgv.jl" -p "5,6,7" -s 100 -ft Float32
```

## Usage information

The accepted command line arguments are (parenthesis for short version):
 - Backend arguments: `--version(-v)`, `--backends(-b)`, `--threads(-t)`. Respectively: Julia version, backend types, number of threads (for Array backend). These arguments accept a list of different parameters, for example:
    ```
    -v "1.8.5 1.9.4" -b "Array CuArray" -t "1 6"
    ```
    which would generate benchmark for all these combinations of parameters.
 - Case arguments: `--cases(-c)`, `--log2p(-p)`, `--max_steps(-s)`, `--ftype(-ft)`. Respectively: Benchmark case file, case sizes, number of time steps, float data type. The following arguments would generate benchmarks for the [`tgv.jl`](./tgv.jl) case:
    ```
    -c "tgv.jl" -p "5,6,7" -s 100 -ft "Float32"
    ```
    which in addition to the benchmark arguments, altogether can be used to launch this script as:
    ```
    sh benchmark.sh -v "1.8.5 1.9.4" -b "Array CuArray" -t "1 6" -c "tgv.jl" -p "5,6,7" -s 100 -ft "Float32"
    ```
    Case arguments accept a list of parameters for each case, and the list index is shared across these arguments (hence lists must have equal length):
    ```
    -c "tgv.jl donut.jl" -p "5,6,7 7,8" -s "100 500" -ft "Float32 Float64"
    ```
    which would run the same benchmarks for the TGV as before, and benchmarks for the donut case too resulting into 2 Julia versions x (2 Array + 1 CuArray) backends x (3 TGV sizes + 2 donut sizes) = 30 benchmarks.

Benchmarks are saved in JSON format with the following nomenclature: `casename_sizes_maxsteps_ftype_backend_waterlilyHEADhash_juliaversion.json`. Benchmarks can be finally compared using [`compare.jl`](./compare.jl) as follows
```
julia --project compare.jl benchmark_1.json benchmark_2.json benchmark_3.json ...
```
Note that each case benchmarks should be compared separately. If a single case is benchmarked, and all the JSON files in the current directory belong to it, one can simply run:
```
julia --project compare.jl $(find . -name "*.json" -printf "%T@ %Tc %p\n" | sort -n | awk '{print $8}')
```
which would take all the JSON files, sort them by creation time, and pass them as arguments to the `compare.jl` program. Finally, note that the first benchmark passed as argument is taken as reference to compute speedups of other benchmarks: `speedup_x = time(benchmark_1) / time(benchmark_x)`.

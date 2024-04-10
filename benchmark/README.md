# Automatic benchmark generation suite

Suite to generate benchmarks across different Julia versions (using [**juliaup**](https://github.com/JuliaLang/juliaup)), backends, cases, and cases sizes using the [benchmark.sh](./benchmark.sh) script.

## TL;DR
Usage example
```sh
sh benchmark.sh  -v "1.9.4 1.10.0" -t "1 6" -b "Array CuArray" -c "tgv jelly" -p "6,7 5,6" -s "100 100" -ft "Float32 Float64"
```
would run both the TGV and jelly benchmarks (`-c`) in 2 different Julia versions (1.9.4 and 1.10.0-rc1, noting that these need to be available in juliaup), and 3 different backends (CPUx1, CPUx6, GPU). The cases size `-p`, number of time steps `-s`, and float type `-ft` are bash (ordered) arrays which need to be equally sized to `-c` and specify each benchmark case (respectively).
The default launch is equivalent to:
```
sh benchmark.sh  -v release -t "1 6" -b "Array CuArray" -c "tgv jelly" -p "6,7 5,6" -s "100 100" -ft "Float32 Float32"
```
Compare each case benchamarks separately using (eg. `tgv`)
```sh
julia --project compare.jl $(find . -name "tgv*.json" -printf "%T@ %Tc %p\n" | sort -n | awk '{print $8}')
```
## Usage information

The accepted command line arguments are (parenthesis for short version):
 - Backend arguments: `--versions(-v)`, `--backends(-b)`, `--threads(-t)`. Respectively: Julia version, backend types, number of threads (when `--backends` contains `Array`). These arguments accept a list of different parameters, for example:
    ```sh
    -v "1.8.5 1.9.4" -b "Array CuArray" -t "1 6"
    ```
    which would generate benchmark for all these combinations of parameters.
 - Case arguments: `--cases(-c)`, `--log2p(-p)`, `--max_steps(-s)`, `--ftype(-ft)`. The `--cases` argument specifies which cases to benchmark, and it can be again a list of different cases. The name of the cases needs to be defined in [benchmark.jl](./benchmark.jl), for example `tgv` or `jelly`. Hence, to add a new case first define the function that returns a `Simulation` in [benchmark.jl](./benchmark.jl), and then it can be called using the `--cases(-c)` list argument. Case size, number of time steps, and float data type are then defined for each case (`-p`, `-s`, `-ft`, respectively). All case arguments must have an equal length since each element of the array defines the case in different aspects.

The following command
```sh
sh benchmark.sh -v release -t "1 3 6" -b "Array CuArray" -c "tgv jelly" -p "6,7,8 5,6" -s "10 100" -ft "Float64 Float32"
```
would allow running benchmarks with 4 backends: CPUx1 (serial), CPUx3, CPUx6, GPU. Additionally, two benchmarks would be tested, `tgv` and `jelly`, with different sizes, number of time steps, and float type, each. This would result into 1 Julia version x (3 Array + 1 CuArray) backends x (3 TGV sizes + 2 jelly sizes) = 20 benchmarks.

Benchmarks are saved in JSON format with the following nomenclature: `casename_sizes_maxsteps_ftype_backend_waterlilyHEADhash_juliaversion.json`. Benchmarks can be finally compared using [`compare.jl`](./compare.jl) as follows
```sh
julia --project compare.jl benchmark_1.json benchmark_2.json benchmark_3.json ...
```
Note that each case benchmarks should be compared separately. If a single case is benchmarked, and all the JSON files in the current directory belong to it, one can simply run (eg. for the `tgv` benchmark):
```sh
julia --project compare.jl $(find . -name "tgv*.json" -printf "%T@ %Tc %p\n" | sort -n | awk '{print $7}') --sort=1
```
which would take all the `tgv` JSON files, sort them by creation time, and pass them as arguments to the `compare.jl` program. Finally, note that the first benchmark passed as argument is taken as reference to compute speedups of other benchmarks: `speedup_x = time(benchmark_1) / time(benchmark_x)`. The `--sort=<1 to 8>` argument can also be used when running the comparison. It will sort the benchmark table rows by the values corresponding to the column index passed as argument. `--sort=1` corresponds to sorting by backend. The baseline row is highlighted in blue, and the fastest run in a table is highlighted in green.

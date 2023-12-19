#!/bin/bash
# ---- Automatic benchmark generation script
# Allows to generate benchmark across different julia versions, backends, cases, and cases sizes.
# juliaup is required: https://github.com/JuliaLang/juliaup
#
# Accepted arguments are (parenthesis for short version):
#   - Backend arguments: --version(-v), --backends(-b) --threads(-t) [Julia version, backend types, number of threads (for Array backend)]
#     These arguments accept a list of different parameters, for example:
#         -v "1.8.5 1.9.4" -b "Array CuArray" -t "1 6"
#     which would generate benchmark for all these combinations of parameters.
#   - Case arguments: --cases(-c), --log2p(-p), --max_steps(-s), --ftype(-ft) [Benchmark case file, case sizes, number of time steps, float data type]
#     The following arguments would generate benchmarks for the "tgv.jl" case:
#         -c "tgv.jl" -p "5,6,7" -s 100 -ft "Float32"
#     which in addition to the benchmark arguments, altogether can be used to launch this script as:
#         sh benchmark.sh -v "1.8.5 1.9.4" -b "Array CuArray" -t "1 3 6" -c "tgv.jl" -p "5,6,7" -s 100 -ft "Float32"
#     Case arguments accept a list of parameters for each case, and the list index is shared across these arguments (hence lists must have equal length):
#         -c "tgv.jl donut.jl" -p "5,6,7 7,8" -s "100 500" -ft "Float32 Float64"
#     which would run the same benchmarks for the TGV as before, and benchmarks for the donut case too resulting into
#         2 Julia versions x (2 Array + 1 CuArray) backends x (3 TGV sizes + 2 donut sizes) = 15 benchmarks
#
# Benchmarks are saved in JSON format with the following nomenclature:
#     casename_sizes_maxsteps_ftype_backend_waterlilyHEADhash_juliaversion.json
# Benchmarks can be finally compared using compare.jl as follows
#     julia --project compare.jl benchmark_1.json benchmark_2.json benchmark_3.json ...
# Note that each case benchmarks should be compared separately.
# If a single case is benchmarked, and all the JSON files in the current directory belong to it, one can simply run:
#     julia --project compare.jl $(find . -name "*.json" -printf "%T@ %Tc %p\n" | sort -n | awk '{print $8}')
# which would take all the JSON files, sort them by creation time, and pass them as arguments to the compare.jl program.
# Finally, note that the first benchmark passed as argument is taken as reference to compute speedups of other benchmarks:
#     speedup_x = time(benchmark_1) / time(benchmark_x).
#
# TL;DR: Usage example
#     sh benchmark.sh  -v "1.9.4 1.10.0-rc1" -t "1 3 6" -b "Array CuArray" -c "tgv.jl" -p "5,6,7"
# The default launch is equivalent to:
#     sh benchmark.sh  -v JULIA_DEFAULT -t "1 6" -b "Array CuArray" -c "tgv.jl" -p "5,6,7" -s 100 -ft Float32
# ----


# Grep current julia version
julia_version () {
    julia_v=($(julia -v))
    echo "${julia_v[2]}"
}

# Update project environment with new Julia version
update_environment () {
    echo "Updating environment to Julia v$version"
    juliaup default $version
    # Mark WaterLily as a development package. Then update dependencies and precompile.
    julia --project -e "using Pkg; Pkg.develop(PackageSpec(path=join(split(pwd(), '/')[1:end-1], '/'))); Pkg.update();"
}

run_benchmark () {
    echo "Running: julia --project $args"
    julia --project $args
}

# Print benchamrks info
display_info () {
    echo "--------------------------------------"
    echo "Running benchmark tests for:
 - Julia:        ${VERSIONS[@]}
 - Backends:     ${BACKENDS[@]}"
    if [[ " ${BACKENDS[*]} " =~ [[:space:]]'Array'[[:space:]] ]]; then
        echo " - CPU threads:  ${THREADS[@]}"
    fi
    echo " - Cases:        ${CASES[@]}
 - Size:         ${LOG2P[@]:0:$NCASES}
 - Sim. steps:   ${MAXSTEPS[@]:0:$NCASES}
 - Data type:    ${FTYPE[@]:0:$NCASES}"
    echo "--------------------------------------"; echo
}

# Default backends
DEFAULT_JULIA_VERSION=$(julia_version)
VERSION=($DEFAULT_JULIA_VERSION)
BACKENDS=('Array' 'CuArray')
THREADS=('1' '6')
# Default cases. Arrays below must be same length (specify each case individually)
CASES=('tgv.jl')
LOG2P=('5,6,7')
MAXSTEPS=('100')
FTYPE=('Float32')

# Parse arguments
while [ $# -gt 0 ]; do
case "$1" in
    --versions|-v)
    VERSIONS=($2)
    shift
    ;;
    --backends|-b)
    BACKENDS=($2)
    shift
    ;;
    --threads|-t)
    THREADS=($2)
    shift
    ;;
    --cases|-c)
    CASES=($2)
    shift
    ;;
    --log2p|-p)
    LOG2P=($2)
    shift
    ;;
    --max_steps|-s)
    MAXSTEPS=($2)
    shift
    ;;
    --float_type|-ft)
    FTYPE=($2)
    shift
    ;;
    *)
    printf "ERROR: Invalid argument\n"
    exit 1
esac
shift
done

NCASES=${#CASES[@]}

# Assert "--threads" argument is not empy if "Array" backend is present
if [[ " ${BACKENDS[*]} " =~ [[:space:]]'Array'[[:space:]] ]]; then
    if [ "${#THREADS[@]}" == 0 ]; then
        echo "ERROR: Backend 'Array' is present, but '--threads' argument is empty."
        exit 1
    fi
fi

# Display information
display_info

# Benchmarks
for version in "${VERSIONS[@]}" ; do
    echo "Julia v$version benchmaks"
    update_environment
    for i in "${!CASES[@]}"; do
        args_case="${CASES[$i]} --log2p="${LOG2P[$i]}" --max_steps=${MAXSTEPS[$i]} --ftype=${FTYPE[$i]}"
        for backend in "${BACKENDS[@]}" ; do
            if [ "${backend}" == "Array" ]; then
                for thread in "${THREADS[@]}" ; do
                    args="-t $thread "$args_case" --backend=$backend"
                    run_benchmark
                done
            else
                args=$args_case" --backend=$backend"
                run_benchmark
            fi
        done
    done
done

# To compare all the benchmarks in this directory, run
# julia --project compare.jl $(find . -name "*.json" -printf "%T@ %Tc %p\n" | sort -n | awk '{print $8}')

# Restore julia system version to default one and exit
juliaup default $DEFAULT_JULIA_VERSION
echo "All done!"
exit 0
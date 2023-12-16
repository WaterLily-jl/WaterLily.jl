#!/bin/bash
# Usage example
# sh launch_bernchmarks.sh  -v "1.8.5 1.10.0-rc1" --threads 6 --backends "Array CuArray" --cases "tgv.jl" --log2n "(3,4)"

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
    echo "Running: julia --projects $args"
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
 - Size:         ${LOG2N[@]}
 - Sim. time:    ${TEND[@]}
 - Max. steps:   ${MAXSTEPS[@]}
 - Data type:    ${DTYPE[@]}
 - Num. samples: ${SAMPLES[@]}"
    echo "--------------------------------------"; echo
}

# Default backends
VERSIONS=($(julia_version))
BACKENDS=('Array' 'CuArray')
THREADS=('1' '6')
# Default cases. Arrays below must be same length (specify each case individually)
CASES=('tgv.jl' 'donut.jl')
LOG2N=('(5,6,7)' '(5,6,7)')
TEND=('10.0' '10.0')
MAXSTEPS=('100' '100')
DTYPE=('Float32' 'Float32')
SAMPLES=('1' '1')

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
    --log2n|-log2n)
    LOG2N=($2)
    shift
    ;;
    --t_end|-tend)
    TEND=($2)
    shift
    ;;
    --max_steps|-maxsteps)
    MAXSTEPS=($2)
    shift
    ;;
    --data_type|-dtype)
    DTYPE=($2)
    shift
    ;;
    --samples|-s)
    SAMPLES=($2)
    shift
    ;;
    *)
    printf "ERROR: Invalid argument\n"
    exit 1
esac
shift
done

# Assert "Array" backend is present if "--threads" argument is passed
if [[ " ${BACKENDS[*]} " =~ [[:space:]]'Array'[[:space:]] ]]; then
    if [ "${#THREADS[@]}" == 0 ]; then
        echo "ERROR: Backend 'Array' present, '--threads' argument is empty."
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
        args_case="${CASES[$i]} --log2n=${LOG2N[$i]} --t_end=${TEND[$i]} --max_steps=${MAXSTEPS[$i]} --dtype=${DTYPE[$i]} --samples=${SAMPLES[$i]}"
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

# Run comparison [ToDo]


# Restore julia system version to default one and exit
juliaup default $(julia_version)
exit 0
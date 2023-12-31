#!/bin/bash

# Grep current julia version
julia_version () {
    julia_v=($(julia -v))
    echo "${julia_v[2]}"
}

# Update project environment with new Julia version
update_environment () {
    echo "Updating environment to Julia v$version"
    # Mark WaterLily as a development package. Then update dependencies and precompile.
    julia +${version} --project -e "using Pkg; Pkg.develop(PackageSpec(path=dirname(@__DIR__))); Pkg.update();"
}

run_benchmark () {
    echo "Running: julia +${version} --project --startup-file=no $args"
    julia +${version} --project --startup-file=no $args
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
JULIA_USER_VERSION=$(julia_version)
VERSIONS=($JULIA_USER_VERSION)
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
    printf "ERROR: Invalid argument %s\n" "${1}" 1>&2
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

echo "All done!"
exit 0

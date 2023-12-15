#!/bin/bash

# Update project environment with new Julia version
update_environment () {
  echo "Updating environment to Julia v$version"
  # juliaup default $version
  julia --project -e "using Pkg; Pkg.update(); Pkg.precompile()"
}

display_info () {
    echo "--------------------------------------"
    echo "Running benchmark tests for:
 - Julia:       ${VERSIONS[@]}
 - Backends:    ${BACKENDS[@]}"
    if [[ " ${BACKENDS[*]} " =~ [[:space:]]'Array'[[:space:]] ]]; then
        echo " - CPU threads: ${THREADS[@]}"
    fi
    echo " - Cases:       ${CASES[@]}
 - Size:        ${LOG2N[@]}
 - Sim. time:   ${TEND[@]}
 - Max. steps:  ${MAXSTEPS[@]}"
    echo "--------------------------------------"; echo
}

# Defaults
# VERSIONS=('1.8.5'  '1.10.0-rc2')
VERSIONS=('1.8.5')
BACKENDS=('Array' 'CuArray')
THREADS=('1' '6')
CASES=('tgv.jl' 'donut.jl')
LOG2N=('(4,5,6)' '(7,8,9)')
TEND=('10.0' '10.0')
MAXSTEPS=('100' '100')

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
    MAXSTEPS=($2)
    shift
    ;;
    --max_steps|-ms)
    MAXSTEPS=($2)
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
    for backend in "${BACKENDS[@]}" ; do
        if [ "${backend}" == "Array" ]; then
            for thread in "${THREADS[@]}" ; do
                args="-t $thread "
            done
        else
            echo "Backend is not Array"
        fi
    done
    # update_environment
    for case in "${CASES[@]}" ; do
        for log2n in "${LOG2N[@]}" ; do

        done
    done
done
#!/bin/bash
set -e
cd "$(dirname "$0")"
julia --project=. -e 'using Pkg; Pkg.instantiate()'
backends="simd ka-cpu"; command -v nvidia-smi >/dev/null && backends="$backends ka-gpu"
for b in $backends; do
    case $b in simd) pref=SIMD; t=1;; ka-cpu) pref=KernelAbstractions; t=auto;; ka-gpu) pref=KernelAbstractions; t=1;; esac
    printf '[WaterLily]\nbackend = "%s"\n' $pref > LocalPreferences.toml
    julia --project=. -t $t bench.jl $b results_$b.csv
done
julia --project=. merge.jl $(for b in $backends; do echo results_$b.csv; done) | tee results.txt

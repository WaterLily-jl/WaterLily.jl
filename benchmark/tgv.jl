using WaterLily
using BenchmarkTools
using CUDA: CuArray
using KernelAbstractions: synchronize, get_backend
using JLD2
using OutMacro

include("util.jl")

log2n, t_end, max_steps, dtype, backend, samples = parse_cla(ARGS;
    log2n=(5,6,7,8), t_end=1.0, max_steps=10, dtype=Float32, backend=Array, samples=5)
evals = 5
verbose = true

function TGV(p, backend; Re=1e5, T=Float32)
    # Define vortex size, velocity, viscosity
    L = 2^p; U = 1; ν = U*L/Re
    # Taylor-Green-Vortex initial velocity field
    function uλ(i,vx)
        x,y,z = @. (vx-1.5)*π/L                # scaled coordinates
        i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x
        i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y
        return 0.                              # u_z
    end
    # Initialize simulation
    return Simulation((L,L,L), (0,0,0), L; U=U, uλ=uλ, ν=ν, T=T, mem=backend)
end

function benchmark()
    suite, results = BenchmarkGroup(), BenchmarkGroup()
    sim_step!(TGV(log2n[1], backend; T=dtype), t_end; max_steps=1, verbose=true, remeasure=false) # warm up
    add_to_suite!(suite, TGV; log2n=log2n, t_end=t_end, max_steps=max_steps, dtype=dtype, backend=backend) # create benchmark
    # tune!(suite)
    results[backend_str[backend]] = run(suite[backend_str[backend]], samples=samples, evals=evals, seconds=1e6, verbose=verbose) # run!
    fname = string(@__DIR__)*"/tgv_simstep_p$(log2n...)_$(backend_str[backend])_v$VERSION.dat"
    save_object(fname, results) # save benchmark
end

benchmark()
using WaterLily
using BenchmarkTools
using CUDA: CuArray
using KernelAbstractions: synchronize, get_backend

include("util.jl")

log2p, max_steps, ftype, backend = parse_cla(ARGS; log2p=(5,6,7,8), max_steps=100, ftype=Float32, backend=Array)

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
    suite = BenchmarkGroup()
    results = BenchmarkGroup(["TGV", "sim_step!", log2p, max_steps, ftype, backend_str[backend], git_hash, string(VERSION)])
    sim_step!(TGV(log2p[1], backend; T=ftype), typemax(ftype); max_steps=1, verbose=false, remeasure=false) # warm up
    add_to_suite!(suite, TGV; log2p=log2p, max_steps=max_steps, ftype=ftype, backend=backend) # create benchmark
    results[backend_str[backend]] = run(suite[backend_str[backend]], samples=1, evals=1, seconds=1e6, verbose=true) # run!
    fname = string(@__DIR__) * "/" *  split(PROGRAM_FILE, '.')[1] *
        "_$(log2p...)_$(max_steps)_$(ftype)_$(backend_str[backend])_$(git_hash)_$VERSION.json"
    BenchmarkTools.save(fname, results)
end

benchmark()
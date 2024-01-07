using WaterLily
using BenchmarkTools
using CUDA: CuArray, allowscalar
using KernelAbstractions: synchronize, get_backend
using StaticArrays

allowscalar(false)
include("util.jl")

cases, log2p, max_steps, ftype, backend = parse_cla(ARGS;
    cases=["tgv", "jelly"], log2p=[(6,7), (5,6)], max_steps=[100, 100], ftype=[Float32, Float32], backend=Array
)

# Define simulation benchmarks
function tgv(p, backend; Re=1e5, T=Float32)
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

function jelly(p, backend; Re=5e2, U=1, T=Float32)
    # Define simulation size, geometry dimensions, & viscosity
    n = 2^p; R = 2n/3; h = 4n-2R; ν = U*R/Re

    # Motion functions
    ω = 2U/R
    @fastmath @inline A(t) = 1 .- SA[1,1,0]*0.1*cos(ω*t)
    @fastmath @inline B(t) = SA[0,0,1]*((cos(ω*t)-1)*R/4-h)
    @fastmath @inline C(t) = SA[0,0,1]*sin(ω*t)*R/4

    # Build jelly from a mapped sphere and plane
    sphere = AutoBody((x,t)->abs(√sum(abs2,x)-R)-1, # sdf
                      (x,t)->A(t).*x+B(t)+C(t))     # map
    plane = AutoBody((x,t)->x[3]-h,(x,t)->x+C(t))
    body =  sphere-plane

    # Return initialized simulation
    Simulation((n,n,4n), (0,0,-U), R; ν, body, T=T, mem=backend)
end

# Generate benchmarks
function benchmark()
    for (case, p, s, ft) in zip(cases, log2p, max_steps, ftype)
        println("Benchmarking: $(case)")
        suite = BenchmarkGroup()
        results = BenchmarkGroup([case, "sim_step!", p, s, ft, backend_str[backend], git_hash, string(VERSION)])
        sim_step!(getf(case)(p[1], backend; T=ft), typemax(ft); max_steps=1, verbose=false, remeasure=false) # warm up
        add_to_suite!(suite, getf(case); p=p, s=s, ft=ft, backend=backend) # create benchmark
        results[backend_str[backend]] = run(suite[backend_str[backend]], samples=1, evals=1, seconds=1e6, verbose=true) # run!
        fname = joinpath(@__DIR__, "$(case)_$(p...)_$(s)_$(ft)_$(backend_str[backend])_$(git_hash)_$VERSION.json")
        BenchmarkTools.save(fname, results)
    end
end

benchmark()
using WaterLily
using BenchmarkTools
using KernelAbstractions: synchronize, get_backend
using StaticArrays
include("util.jl")

# Define simulation benchmarks
function tgv(p, backend; Re=1600, T=Float32)
    L = 2^p; U = 1; κ=π/L; ν = 1/(κ*Re)
    function uλ(i,xyz)
        x,y,z = @. xyz/L*π
        i==1 && return -U*sin(x)*cos(y)*cos(z)
        i==2 && return  U*cos(x)*sin(y)*cos(z)
        return 0.
    end
    Simulation((L, L, L), (0, 0, 0), 1/κ; U=U, uλ=uλ, ν=ν, T=T, mem=backend)
end

function sphere(p, backend; Re=3700, U=1, T=Float32)
    D = 2^p; ν = U*D/Re
    L = (16D, 6D, 6D)
    center = @SVector T[1.5D, 3D, 3D]; radius = T(D/2)
    body = AutoBody((x,t) -> √sum(abs2, x .- center) - radius)
    Simulation(L, (U, 0, 0), D; U=U, ν=ν, body=body, T=T, mem=backend, exitBC=true)
end

function cylinder(p, backend; Re=1e3, U=1, T=Float32)
    L = 2^p; R = L/2; ν = U*L/Re
    center = SA[1.5L, 3L, 0]
    function sdf(xyz, t)
        x, y, z = xyz - center
        √sum(abs2, SA[x, y, 0]) - R
    end
    function map(xyz, t)
        xyz - SA[0, R*sin(t*U/L), 0]
    end
    Simulation((12L, 6L, 2L), (U, 0, 0), L; U=U, ν=ν, body=AutoBody(sdf, map), T=T, mem=backend, exitBC=true, perdir=(3,))
end

function donut(p, backend; Re=1e3, U=1, T=Float32)
    L = 2^p
    center, R, r = SA[L/2, L/2, L/2], L/4, L/16
    ν = U*R/Re
    norm2(x) = √sum(abs2,x)
    body = AutoBody() do xyz, t
        x, y, z = xyz - center
        norm2(SA[x, norm2(SA[y, z]) - R]) - r
    end
    Simulation((2L, L, L), (U, 0, 0), R; ν, body, T=T, mem=backend)
end

function jelly(p, backend; Re=5e2, U=1, T=Float32)
    n = 2^p; R = 2n/3; h = 4n - 2R; ν = U*R/Re
    ω = 2U/R
    @fastmath @inline A(t) = 1 .- SA[1,1,0]*0.1*cos(ω*t)
    @fastmath @inline B(t) = SA[0,0,1]*((cos(ω*t) - 1)*R/4-h)
    @fastmath @inline C(t) = SA[0,0,1]*sin(ω*t)*R/4
    sphere = AutoBody((x,t)->abs(√sum(abs2, x) - R) - 1,
                      (x,t)->A(t).*x + B(t) + C(t))
    plane = AutoBody((x,t)->x[3] - h, (x, t) -> x + C(t))
    body =  sphere - plane
    Simulation((n, n, 4n), (0, 0, -U), R; ν, body, T=T, mem=backend)
end

# Generate benchmarks
function run_benchmarks(cases, log2p, max_steps, ftype, backend, bstr; datadir="./")
    for (case, p, s, ft) in zip(cases, log2p, max_steps, ftype)
        println("Benchmarking: $(case)")
        suite = BenchmarkGroup()
        results = BenchmarkGroup([case, "sim_step!", p, s, ft, bstr, git_hash, string(VERSION)])
        add_to_suite!(suite, getf(case); p=p, s=s, ft=ft, backend=backend, bstr=bstr,
            remeasure=any(occursin.(["cylinder", "jelly"], case))
        ) # create benchmark
        results[bstr] = run(suite[bstr], samples=1, evals=1, seconds=1e6, verbose=true) # run!
        fname = "$(case)_$(p...)_$(s)_$(ft)_$(bstr)_$(git_hash)_$VERSION.json"
        BenchmarkTools.save(joinpath(datadir,fname), results)
    end
end

cases, log2p, max_steps, ftype, backend = parse_cla(ARGS;
    cases=["tgv", "jelly"], log2p=[(6,7), (5,6)], max_steps=[100, 100], ftype=[Float32, Float32], backend=Array
)

# Generate benchmark data
datadir = joinpath("data", git_hash)
mkpath(datadir)
run_benchmarks(cases, log2p, max_steps, ftype, backend, backend_str[backend]; datadir)
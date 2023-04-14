using WaterLily

# Set-up and quick sim check
function sphere_sim(radius = 8; Re = 250, T=Float32, mem=Array, domain::NTuple{N} = (6,4)) where N
    body = AutoBody((x,t)-> √sum(abs2,x .- 2radius) - radius)
    n = map(d->d*radius, domain)
    U = δ(1,N).I
    return Simulation(n,U,radius; body, ν=radius/Re, T, mem)
end
function sphere_example(radius=8,twoD=true)
    domain = twoD ? (6,4) : (6,2,2)
    sphere_sim(radius;domain)
end
# begin
#     sim = sphere_example(16,true);
#     mom_step!(sim.flow,sim.pois)
# end

# Profile
function profile_sim(sim,its=2^28÷length(sim.flow.u))
    for t=1:its
        mom_step!(sim.flow,sim.pois)
    end
end
# begin
#     sim = sphere_example(16,false);
#     @profview profile_sim(sim,1)
#     @profview profile_sim(sim)        
# end

# Benchmark
using BenchmarkTools,DataFrames,CSV
function benchmark_sim(sim)
    sim_step!(sim,0.1)
    WaterLily.DISABLE_PUSH()
    a = @benchmark mom_step!(sim2.flow,sim2.pois) setup=(sim2=$sim) seconds=20 samples=1000
    WaterLily.ENABLE_PUSH()
    a = minimum(a)
    b = Dict(key => getfield(a, key) for key in propertynames(a))
    b[:n] = length(sim.flow.u)
    return b
end
function create_CSV(N,twoD)
    df = DataFrame((@show n;benchmark_sim(sphere_example(n,twoD))) for n ∈ N)
    select!(df,Not([:params]))
    CSV.write("benchmark\\threads"*string(Threads.nthreads())*"_2D"*string(twoD)*".csv",df)
end
create_CSV(2 .^ (3:8),true)
create_CSV(2 .^ (3:6),false)

begin
    using Plots
    plot(xaxis=("length u",:log10), yaxis=("time (ns)",:log10))
    df = DataFrame(CSV.File("benchmark\\threads1_2Dtrue.csv"))
    scatter!(df.n,df.time,label="2D, single threaded")
    df = DataFrame(CSV.File("benchmark\\threads20_2Dtrue.csv"))
    scatter!(df.n,df.time,label="2D, multi-threaded")
    df = DataFrame(CSV.File("benchmark\\threads1_2Dfalse.csv"))
    scatter!(df.n,df.time,label="3D, single threaded")
    df = DataFrame(CSV.File("benchmark\\threads20_2Dfalse.csv"))
    scatter!(df.n,df.time,label="3D, multi-threaded")
    savefig("benchmark\\benchmark.png")
end
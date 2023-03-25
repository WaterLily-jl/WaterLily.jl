using WaterLily

# Set-up and quick sim check
function sphere_sim(radius = 8; Re = 250, T=Float64, domain = (6,4))
    body = AutoBody((x,t)-> √sum(abs2,x .- 2radius) - radius)
    n = map(d->d*radius+2, domain)
    U = zeros(T,length(domain)); U[1] = 1
    return Simulation(n,U,radius; body, ν=U[1]*radius/Re,T)
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
using BenchmarkTools
function benchmark_sim(sim)
    sim_step!(sim,1)
    WaterLily.DISABLE_PUSH()
    @btime mom_step!(sim2.flow,sim2.pois) setup=(sim2=$sim) seconds=20
    WaterLily.ENABLE_PUSH()
end
begin
    sim = sphere_example(16,false);
    benchmark_sim(sim)
end

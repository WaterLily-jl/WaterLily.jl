include("../src/WaterLily.jl")
using .WaterLily
using LinearAlgebra: norm2
using JLD2

function circle(radius=8;Re=250,n=10,m=6)
    center, ν = radius*m/2, radius/Re
    body = AutoBody((x,t)->norm2(x .- center) - radius)
    Simulation((n*radius+2,m*radius+2), [1.,0.], radius; ν, body)
end

function sim_integrate!(sim;duration=1,step=0.1,verbose=false,
                        remeasure=false,kv...)
    t₀ = round(sim_time(sim))
    t = range(t₀,t₀+duration;step)
    duration = @elapsed for tᵢ in t
    sim_step!(sim,tᵢ;remeasure)
    verbose && println("tU/L=",round(tᵢ,digits=4),
    ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
    return duration
    end

sim = circle(16)
runtime = sim_integrate!(sim;duration=1,step=0.25)
filename_u = string(tempdir(), "/testing_using_threads#",
                    Threads.nthreads(), "_u.jld2")
save(filename_u, "data", sim.flow.u)
# filename_t = string(tempdir(), "\\testing_using_threads#",
#                     Threads.nthreads(), "_t.jld")
# save(filename_t, "data", runtime)
@info("Saved final velocity field to ", filename_u)

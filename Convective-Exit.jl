using WaterLily,SpecialFunctions,ForwardDiff,StaticArrays

function lamb_dipole(N;D=N/3,U=1,exit=false)
    β = 2.4394π/D
    @show besselj1(β*D/2)
    C = -2U/(β*besselj0(β*D/2))
    function ψ(x,y)
        r = √(x^2+y^2)
        ifelse(r ≥ D/2, U*((D/2r)^2-1)*y, C*besselj1(β*r)*y/r)
    end
    center = SA[N/2,N/2]
    function uλ(i,xy)
        x,y = xy-center
        ifelse(i==1,ForwardDiff.derivative(y->ψ(x,y),y)+1+U,-ForwardDiff.derivative(x->ψ(x,y),x))
    end
    Simulation((N, N), (1,0), D; uλ, exit)
end

include("examples/TwoD_plots.jl")

sim = lamb_dipole(64,U=0.25);
@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
flood(sim.flow.σ,clims=(-5,5))
sim_step!(sim,1.2,remeasure=false)
@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
flood(sim.flow.σ,clims=(-5,5))

sim = lamb_dipole(64,U=0.25,exit=true);
sim_step!(sim,1.2,remeasure=false)
@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
flood(sim.flow.σ,clims=(-5,5))

U=2
sim = lamb_dipole(64,U=U,exit=true);
sim_gif!(sim,duration=2.5/(1+U),step=0.05,clims=(-20,20))
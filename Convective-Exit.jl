using WaterLily
using BenchmarkTools
include("examples/TwoD_plots.jl")

function test_conv_BC!(sim)
    # duration of the simulation
    duration = 16
    step = 0.4
    t₀ = 0.0
    plot()

    @time @gif for tᵢ in range(t₀,t₀+duration;step)

        # update until time tᵢ in the background
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U

            # update flow
            # mom_step_BC!(sim.flow,sim.pois)
            WaterLily.mom_step!(sim.flow,sim.pois)
                    
            # compute motion and acceleration 1DOF
            Δt = sim.flow.Δt[end]
            t += Δt
        end

        # plot vorticity
        # @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        # flood(sim.flow.σ; shift=(-0.5,-0.5), clims=(-5,5))
        # flood(sim.flow.p; shift=(-0.5,-0.5), clims=(-1,1))
        # addbody(real(z.+n/4),imag(z.+im*m/2))
        plot!(sim.flow.u[end,:,1])
        # plot!(sim.flow.u[end,:,1])

        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

L = 4
N = (L,L)
Nd = N .+ 2
Nd = (Nd...,2)
U = (1.0,0.0)
u = Array{Float64}(undef, Nd...)

# vertical shear
println("--Before - BC(u,U)--")
apply!((i, x) -> (i==1 ? x[2] : 0), u)
BC!(u,zeros(2))
display(u[:,:,1]')

println("--BCΔt!(u,U;Δt=1)--")
BC!(u,zeros(2)) # reset BC values
WaterLily.BCΔt!(u, (1,0); Δt=1.0) # convect and mass check
display(u[:,:,1]')

println("--BCΔt!(u,U)--") # same as BC!(u,1)
BC!(u,zeros(2)) # reset BC values
WaterLily.BCΔt!(u, (1,0)) # only mass check
display(u[:,:,1]')

# test on flow sim
# some definitons
U = 1
Re = 250

m, n = 32,48
println("$n x $m: ", prod((m,n)))
radius = 4

# make a circle body
body = AutoBody((x,t)->√sum(abs2, x .- [n/4,m/2]) - radius)
z = [radius*exp(im*θ) for θ ∈ range(0,2π,length=33)]

# make a simulation
sim = Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, T=Float64)
test_conv_BC!(sim)

# flood(sim.flow.u[2:end-1,2:end-1,1])
# flood(sim.flow.p[2:end-1,2:end-1])
# flood(sim.flow.μ₀[:,:,1])
# plot(sim.flow.u[end,:,1])
# plot!(sim.flow.u[end-1,:,1])


# test all contributions to mom_step!
# @benchmark WaterLily.mom_step!(sim.flow,sim.pois)

# # step by step
# println("conv_diff!")
# @benchmark WaterLily.conv_diff!(sim.flow.f,sim.flow.u⁰,sim.flow.σ,ν=sim.flow.ν)

# println("BDIM!")
# @benchmark WaterLily.BDIM!(sim.flow)

# println("BCΔt!")
# @benchmark WaterLily.BCΔt!(sim.flow.u,sim.flow.U;Δt=sim.flow.Δt[end]) 

# println("project!")
# @benchmark WaterLily.project!(sim.flow,sim.pois)

# println("BCΔt!")
# @benchmark WaterLily.BC!(sim.flow.u,sim.flow.U)



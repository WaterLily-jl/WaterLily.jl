using WaterLily
using Plots; gr()
using StaticArrays
include("TwoD_plots.jl")

# required to keep things global
let 
    # parameters
    Re = 250; U = 1
    p = 5; L = 2^p
    radius, center = L/2, 2L

    # fsi parameters
    T = 4*radius    # VIV period
    mₐ = π*radius^2 # added-mass coefficent circle
    m = 0.1*mₐ      # mass as a fraction of the added-mass, can be zero
    k = (2*pi/T)^2*(m+mₐ)

    # initial condition FSI
    p0=radius/3; v0=0; a0=0; t0=0

    # motion function uses global var to adjust
    posx(t) = p0 + (t-t0)*v0

    # motion definition
    map(x,t) = x - SA[0, posx(t)]

    # mₐke a body
    circle = AutoBody((x,t)->√sum(abs2, x .- center) - radius, map)

    # generate sim
    sim = Simulation((6L,4L), (U,0), radius; ν=U*radius/Re, body=circle)

    # get start time
    duration=10; step=0.1; t₀=round(sim_time(sim))

    @time @gif for tᵢ in range(t₀,t₀+duration;step)

        # update until time tᵢ in the background
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U

            # measure body
            measure!(sim,t)

            # update flow
            mom_step!(sim.flow,sim.pois)
            
            # pressure force
            force = -WaterLily.∮nds(sim.flow.p,sim.flow.f,sim.body,t)
            
            # compute motion and acceleration 1DOF
            Δt = sim.flow.Δt[end]
            accel = (force[2]- k*p0 + mₐ*a0)/(m + mₐ)
            p0 += Δt*(v0+Δt*accel/2.) 
            v0 += Δt*accel
            a0 = accel

            # update time, sets the pos/v0 correctly
            t0 = t; t += Δt
        end

        # plot vorticity
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ; shift=(-0.5,-0.5),clims=(-5,5))
        body_plot!(sim); plot!(title="tU/L $tᵢ")
        
        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end
using WaterLily
using Plots; gr()
using StaticArrays
include("TwoD_plots.jl")

# parameters
Re = 250
U = 1
nx, ny = 3*2^6, 2^7
radius, center = ny/8, ny/2

# fsi parameters
T = 10*radius
amp = 0.354
mₐ = π*radius^2  # added-mass coefficent circle http://brennen.caltech.edu/fluidbook/basicfluiddynamics/unsteadyflows/addedmass/valuesoftheaddedmass.pdf
m = 0.0*mₐ       # zero actual mass
k = (2*pi/T)^2*(m+mₐ)

# initial conditions
pos = amp*20
vel = 0; a0 = 0; t_init = 0

# motion function uses global var to adjustset the velocity
# to the one computed from the FSI, this is because the velocity
# is computed internaly using ForwardDiff
y(t) = pos + (t-t_init)*vel
function map(x,t)
    x - SA[0,y(t)]
end

# make a body
circle = AutoBody((x,t)->√sum(abs2, x .- center) - radius, map)

# generate sim
sim = Simulation((nx,ny), (U,0), radius; ν=U*radius/Re, body=circle)

# get start time
duration = 20; step = 0.1
t₀ = round(sim_time(sim))
@time @gif for tᵢ in range(t₀,t₀+duration;step)

    # update until time tᵢ in the background, equivalent to
    # sim_step!(sim,tᵢ;remeasure=true)
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U

        # measure body
        measure!(sim,t)

        # update flow
        mom_step!(sim.flow,sim.pois)
        
        # pressure force
        force = -WaterLily.∮nds(sim.flow.p,sim.flow.f,circle,t)
        
        # compute motion and acceleration 1DOF
        Δt = sim.flow.Δt[end]
        accel = (force[2]- k*pos + mₐ*a0)/(m + mₐ)
        global pos += Δt*(vel+Δt*accel/2.) 
        global vel += Δt*accel
        global a0 = accel
        
        # update time, must be done globaly to set the pos/vel correctly
        global t_init = t
        t += Δt
    end

    # plot vorticity
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ; shift=(-0.5,-0.5),clims=(-5,5))
    body_plot!(sim)
    
    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end

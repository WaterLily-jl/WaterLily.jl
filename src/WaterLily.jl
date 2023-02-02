module WaterLily

_nthread = Threads.nthreads()
if _nthread==1
    @warn "WaterLily.jl is running on a single thread.\n
Launch Julia with multiple threads to enable multithreaded capabilities:\n
    \$julia -t auto $PROGRAM_FILE"
else
    print("WaterLily.jl is running on ", _nthread, " thread(s)\n")
end

include("util.jl")
export L₂,BC!,@inside,inside,δ,apply!,loc

include("Poisson.jl")
export AbstractPoisson,Poisson,solver!,mult

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solver!,mult

include("Flow.jl")
export Flow,mom_step!

include("Body.jl")
export AbstractBody

include("AutoBody.jl")
export AutoBody,measure!,measure

include("Metrics.jl")
using LinearAlgebra: norm2

"""
    Simulation(dims::Tuple, u_BC::Vector, L::Number;
               U=norm2(u_BC), Δt=0.25, ν=0., ϵ = 1,
               uλ::Function=(i,x)->u_BC[i],
               body::AbstractBody=NoBody())

Constructor for a WaterLily.jl simulation:

    `dims`: Simulation domain dimensions.
    `u_BC`: Simulation domain velocity boundary conditions, `u_BC[i]=uᵢ, i=1,2...`.
    `L`: Simulation length scale.
    `U`: Simulation velocity scale.
    `ϵ`: BDIM kernel width.
    `Δt`: Initial time step.
    `ν`: Scaled viscosity (`Re=UL/ν`)
    `uλ`: Function to generate the initial velocity field.
    `body`: Immersed geometry

See files in `examples` folder for examples.
"""
struct Simulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    function Simulation(dims::Tuple, u_BC::Vector, L::Number;
                        Δt=0.25, ν=0., U=norm2(u_BC), ϵ = 1,
                        uλ::Function=(i,x)->u_BC[i],
                        body::AbstractBody=NoBody(),T=Float64)
        flow = Flow(dims,u_BC;uλ,Δt,ν,T)
        measure!(flow,body;ϵ)
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.μ₀))
    end
end

time(sim::Simulation) = sum(sim.flow.Δt[1:end-1])
"""
    sim_time(sim::Simulation)

Return the current dimensionless time of the simulation `tU/L`
where `t=sum(Δt)`, and `U`,`L` are the simulation velocity and length
scales.
"""
sim_time(sim::Simulation) = time(sim)*sim.U/sim.L

"""
    sim_step!(sim::Simulation,t_end;verbose=false)

Integrate the simulation `sim` up to dimensionless time `t_end`.
If `verbose=true` the time `tU/L` and adaptive time step `Δt` are
printed every time step.
"""
function sim_step!(sim::Simulation,t_end;verbose=false,remeasure=false)
    t = time(sim)
    while t < t_end*sim.L/sim.U
        remeasure && measure!(sim,t)
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
        verbose && println("tU/L=",round(t*sim.U/sim.L,digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

"""
    measure!(sim::Simulation,t=time(sim))

Measure a dynamic `body` to update the `flow` and `pois` coefficients.
"""
function measure!(sim::Simulation,t=time(sim))
    measure!(sim.flow,sim.body;t,ϵ=sim.ϵ)
    update!(sim.pois,sim.flow.μ₀)
end

@fastmath kern₀(d) = 0.5+0.5d+0.5sin(π*d)/π
μ₀(d,ϵ) = kern₀(clamp(d/ϵ,-1,1))

function offset(Bodies, L)
	"""Creates a different offset for each of the bodies so that when the general map is computed, there is no problem of
	body spontaneous generation. This offset is then substracted when the individual map is applied where it needs to be thus
	hiding its existence."""
	
    sdfList = [ (x,t) -> offsetSdf(x,t,i) for i in 1:length(Bodies)]
    mapList = [ (x,t) -> offsetMap(x,t,i) for i in 1:length(Bodies)]

	function offsetSdf(x,t, i)
		xc = x + [0.,(-1)^i * 100*i * L]
		return Bodies[i][1](xc,t)
	end

	function offsetMap(x,t, i)
		xc = x - [0.,(-1)^i * 100*i * L]
		return Bodies[i][2](xc,t)
	end

	return (sdfList, mapList)
end

function addBody(Bodies::Array{SVector{Function, Function}}, L=100)
	"""addBody(Bodies::Array{SVector{Function, Function}}, L=100)
    
        Bodies: array of SVector(sdf, map) for each of the independent bodies to add to the window.
        L: carateristic dimension of the largest body, to create a great enough offset to delete the generetion of undesired body.

        
    The default distance between two independent bodies is set to 100L. It impacts both their placement to not disturbe the other maps,
	in the function 'offset', and the selection of the second closest body to a given point in the function 'min_excluding_i'.
	The coefficients are computed to determine where each map should be used, therefore creating a global map that impacts the 
	whole simulation window.

	The output can directly be used as the body argument in the Simulation function provided by WaterLily."""

	sdfList, mapList = offset(Bodies, L)

	function min_excluding_i(sdfL, mapL, i, x, t)
		min_val = 100L
		for j in eachindex(sdfL)
			if j != i 
				val = sdfL[j](mapL[j](x,t),t)
				if val <= min_val
					min_val = val
				end
			end
		end
		return min_val
	end

	coef = [(x,t) -> μ₀(min_excluding_i(sdfList, mapList, i, x, t) - sdfList[i](mapList[i](x,t),t),1) for i in range(1, length(sdfList))]

	sdf(x,t) = minimum([sdfX(x,t) for sdfX in sdfList])
	map(x,t) = sum([mapList[i](x,t)*coef[i](x,t) for i in range(1, length(mapList))])

	return AutoBody(sdf, map)
end

export Simulation,sim_step!,sim_time,measure!, addBody
end # module

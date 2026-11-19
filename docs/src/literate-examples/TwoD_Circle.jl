# # [Flow over a circle](@id examples-TwoD_Circle)
#
# ![](TwoD_Circle.gif)
#
# *Figure 1*: Vorticity ``\omega L/U`` behind a circular cylinder at ``Re=UR/\nu=250``, based
# on the radius ``R`` (``Re_D=500`` on the diameter). Red is counter-clockwise rotation, blue
# is clockwise, and the black line is the zero contour of the body's signed distance function.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`TwoD_Circle.ipynb`](TwoD_Circle.ipynb).
#-
#
# ## Introduction
#
# This is the "Hello, world!" of WaterLily: the flow over a circular cylinder, the problem
# every immersed-boundary solver is measured against. Above ``Re_D \approx 50`` the wake
# becomes unstable and sheds vortices alternately from each side, producing the von Kármán
# vortex street in the figure and an oscillating side force on the body.
#
# Everything in WaterLily is built out of the three pieces this example introduces:
#
#  1. a **geometry**, defined implicitly by a signed distance function and wrapped in an
#     [`AutoBody`](@ref),
#  2. a **`Simulation`**, which holds the grid, the flow fields and the pressure solver,
#  3. a **time loop** driven by [`sim_step!`](@ref), between whose steps you measure and
#     plot whatever you like.
#
# Every other example on these pages is a variation on these three pieces, so it is worth
# reading this one carefully.
#
# ## Commented program
#
# What follows is the program, spliced with comments.
#md # The full program, without comments, can be found in the next
#md # [section](@ref TwoD_Circle-plain-program).
#
# We need WaterLily itself, and Plots for the figures. Loading Plots also activates
# WaterLily's plotting extension, which is where `flood`, `body_plot!` and
# `sim_gif!` come from.

using WaterLily, Plots

# ### Defining the geometry
#
# A body is defined *implicitly*, by a function returning the signed distance from a point
# `x` to the body surface: negative inside, positive outside, zero on the surface. For a
# circle of radius `r` centred at `c` that is simply ``|x-c| - r``.
#
# `AutoBody` wraps the distance function and uses automatic differentiation to work out
# everything else it needs: the surface normal and, if the body moves, its velocity. You
# never have to supply a mesh, a panel distribution or a normal vector.
#
# Note the length and velocity *scales* passed to `Simulation`: WaterLily is
# non-dimensional, so the viscosity is set as `ν = U*L/Re` and all times reported by
# [`sim_time`](@ref) are in units of ``L/U``. Here the length scale `L` is the *radius*, so
# `Re=250` is ``Re_D=500`` based on the diameter, the convention most reference data uses.

function circle(n, m; Re = 250, U = 1, mem = Array, T = Float32)
    radius, center = m / 8, m / 2
    body = AutoBody((x, t) -> √sum(abs2, x .- center) - radius)
    Simulation((n, m), (U, 0), radius; U, ν = U * radius / Re, body, mem, T)
end

# Now build a simulation on a ``192 \times 128`` grid. The circle diameter is `m/4 = 32`
# cells, which is on the coarse side but plenty to get the shedding right; the resolution is
# kept small here so that these docs build quickly. Pass `mem=CuArray` (after
# `using CUDA`) to run exactly the same simulation on an NVIDIA GPU.

sim = circle(3 * 2^6, 2^7);

# ### The first step, and a check against potential flow
#
# `sim.flow` holds the flow state. The velocity `sim.flow.u` is a staggered (face-centred)
# field of size `(n+2, m+2, 2)`, so `sim.flow.u[:,:,1]` is the ``x``-component. Straight after
# construction it is the uniform free stream *everywhere* -- the body has been measured onto
# the grid, but it has not been imposed on the flow yet, so you cannot see it.
#
# One call to [`sim_step!`](@ref) with no end time takes a single time step, which applies the
# Boundary Data Immersion Method forcing and the pressure projection. Now the body appears:

sim_step!(sim)
contourf(sim.flow.u[:, :, 1]' |> Array, aspect_ratio = :equal, levels = 20,
    title = "u/U after one step", linewidth = 0)

# That first step is worth looking at, because the wake has had no time to form and the result
# should be close to the classical **potential flow** around a cylinder. That solution has
# stagnation points fore and aft, and accelerates the flow to exactly ``2U`` over the shoulder --
# an easy thing to check:

c = size(sim.flow.p, 2) ÷ 2                       # centreline row
u_shoulder = sim.flow.u[c, c + round(Int, sim.L) + 1, 1] # just above the top of the cylinder
"u/U over the shoulder = $(round(u_shoulder, digits = 3))   (potential flow: 2)"

# A few percent low on a 32-cell diameter, which is about what a first-order immersed boundary
# should give here. If you change the geometry or the boundary conditions and the first step does
# *not* look like potential flow, something is wrong before any of the interesting physics has
# started.
#
# ### Stepping the flow forward
#
# [`sim_step!`](@ref)`(sim, t)` advances the simulation until `sim_time(sim) ≥ t`, taking as
# many internal time steps as the CFL condition requires, so you control the *output* interval
# and WaterLily controls the time step. Because this body never moves we pass
# `remeasure=false`, which skips re-measuring the geometry each step.
#
# Between steps we compute the vorticity ``\omega = \nabla \times u`` into the scratch array
# `sim.flow.σ` with the [`@inside`](@ref) macro (WaterLily's loop over the interior cells of
# an array) and flood-plot it. `inside(sim.flow.p)` is the range of cells excluding the
# ghost cells at the domain edge.

# We record the force at the same time, so one pass through the flow gives both the animation
# and the history plotted below. `WaterLily.pressure_force` integrates ``p\,n`` over the immersed
# surface with `n` the **outward** normal, so what it returns is the force the body exerts on the
# fluid. Flip the sign to get the force *on the body* -- the drag and lift you actually want --
# and scale by ``\tfrac{1}{2}\rho U^2 L`` (with ``\rho = 1``) to make it a coefficient.

force_coefficients(sim) = -WaterLily.pressure_force(sim) ./ (0.5 * sim.L * sim.U^2)

R = inside(sim.flow.p)
times = range(0, 40; step = 0.2)
forces = Vector{Float64}[]
anim = @animate for tᵢ in times
    sim_step!(sim, tᵢ; remeasure = false)
    push!(forces, force_coefficients(sim))
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
    flood(sim.flow.σ[R]; clims = (-10, 10), cfill = :seismic, legend = false, border = :none)
    body_plot!(sim)
end
gif(anim, "TwoD_Circle.gif", fps = 24)
nothing #hide

# The animation at the top of the page is the result. If you do not need anything custom in the
# frames, WaterLily bundles the stepping, the vorticity and the plotting into one call --
# `sim_gif!(sim; duration=40, clims=(-10,10), plotbody=true)` -- but then there is nowhere to
# hang the force measurement, which is why the loop is written out here.
#
# ### The forces
#
# Vorticity pictures are pretty, but the reason to run a simulation is usually a number. Plotting
# the history recorded above, from rest, puts the whole story in one figure:

plot(times, [first.(forces) last.(forces)], labels = ["drag" "lift"],
    xlabel = "tU/L", ylabel = "Pressure force coefficient")

# The drag starts very high -- an impulsively started body has to accelerate the fluid around it
# -- and then settles. The lift stays near zero while the wake is still symmetric, and only once
# the wake buckles does it grow into the oscillation that gives the vortex street away. Lift
# oscillates at the shedding frequency and drag at twice it, because each shed vortex pushes the
# cylinder sideways once but fore-and-aft twice per cycle.
#
# The mean drag is the standard number to validate against. `sim.L` here is the cylinder
# *radius*, so halving converts to the usual diameter-based coefficient, to be compared at the
# diameter-based ``Re_D=500``:

settled = first.(forces)[length(forces) ÷ 2:end]
"Cd (diameter-based) = $(round(sum(settled) / length(settled) / 2, digits = 3))   (measured at Re_D=500: about 1.1 to 1.2)"

# That is in the range of the measured values, on a grid this coarse, although the agreement is
# partly fortunate: this is the pressure drag alone, and a 2D simulation has none of the
# three-dimensional wake that a real cylinder develops at this Reynolds number. Raising the
# resolution and running longer tightens up both this and the Strouhal number.
#
# ### Where to go from here
#
# Swap the distance function in `circle` for any other and you are simulating the flow around
# something else; that is the whole point of an implicit geometry. The
# [next example](@ref examples-TwoD_Square) builds a sharp-edged body out of several distance
# functions at once.

#md # ## [Plain program](@id TwoD_Circle-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`TwoD_Circle.jl`](TwoD_Circle.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```

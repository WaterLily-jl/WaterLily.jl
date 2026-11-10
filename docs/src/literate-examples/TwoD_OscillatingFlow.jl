# # [Oscillating flow over a circle](@id examples-TwoD_OscillatingFlow)
#
# ![](TwoD_OscillatingFlow.gif)
#
# *Figure 1*: A circular cylinder in a flow that reverses every half period. Vorticity shed in
# one half cycle is swept back over the body in the next.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`TwoD_OscillatingFlow.ipynb`](TwoD_OscillatingFlow.ipynb).
#-
#
# ## Introduction
#
# Oscillating flow over a cylinder is the canonical model problem for wave loading on a
# structure, and it is governed by the Keulegan–Carpenter number ``KC = U T / D``, how far the
# fluid travels in one period compared to the body size. For small ``KC`` the vorticity shed in
# one half cycle never escapes and is re-ingested in the next, which is why the wake looks
# nothing like a vortex street.
#
# The [previous example](@ref examples-TwoD_SlowStartCircle) drove the flow from the boundary.
# That works, but it is not how you would set up an oscillating flow: a time-dependent inflow
# at the boundary has to travel into the domain, and the pressure field has to adjust
# continuously to a boundary condition that keeps changing sign. The cleaner formulation is to
# move into the frame of the oscillating fluid and drive the flow with a **uniform body
# force**, the `g` keyword:
#
# ```math
# \frac{\partial u}{\partial t} + u \cdot \nabla u = -\nabla p + \nu \nabla^2 u + g(t)
# ```
#
# where `g(i, x, t)` returns the ``i``-th component of ``du_i/dt``. This example also
# introduces `perdir`, which makes a domain direction **periodic** instead of inflow/outflow,
# the natural choice when the mean flow is zero and there is no "upstream".
#
# ## Commented program
#
#md # The full program, without comments, can be found in the next
#md # [section](@ref TwoD_OscillatingFlow-plain-program).

using WaterLily, Plots

# ### Driving the flow with a body force
#
# We pick the oscillation from `κ`, the ratio of the oscillation length scale to the radius.
# The acceleration amplitude and the time scale follow from it, and `g` is then a pure sine in
# the streamwise direction. As with a `uBC` function, `g` must be type stable and return
# `zero(T)` for the components it does not drive.
#
# `perdir=(1,)` makes the first (streamwise) direction periodic. The flow leaving the right of
# the domain re-enters on the left, so there is no exit boundary for the reversing flow to
# fight with.

function circle(n, m; κ = 1.5, Re = 250, U = 1, mem = Array, T = Float32)
    radius = T(m / 8)
    U, κ = T(U), T(κ)
    body = AutoBody((x, t) -> √sum(abs2, x .- (n / 2, m / 2)) - radius)
    ## time-varying uniform acceleration
    accelScale, timeScale = U^2 / 2radius, κ * radius / U
    g(i, x, t::T) where {T} = i == 1 ? -2accelScale * sin(t / timeScale) : zero(T)
    Simulation((n, m), (U, zero(T)), radius; U, ν = U * radius / Re, body, g,
        perdir = (1,), mem, T)
end

sim = circle(2^7, 2^7);

# ### Running it
#
# Nothing about the time loop changes: a body force is part of the momentum equation, not
# something you have to drive by hand. We suppress the tiny residual vorticity before plotting
# so that the quiescent fluid stays white rather than speckled.

R = inside(sim.flow.p)
anim = @animate for tᵢ in range(0, 20; step = 0.2)
    sim_step!(sim, tᵢ; remeasure = false)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0f0, sim.flow.σ[I])
    flood(sim.flow.σ[R]; clims = (-8, 8), cfill = :seismic, legend = false, border = :none)
    body_plot!(sim)
end
gif(anim, "TwoD_OscillatingFlow.gif", fps = 24)
nothing #hide

# ### The in-line force
#
# The force on the body is the quantity of engineering interest, and in oscillating flow it
# splits into a drag component in phase with the velocity and an inertia component in phase
# with the acceleration (Morison's equation). Here we just record the in-line pressure force
# over a couple of periods, against the driving acceleration.

function inline_force!(sim, t)
    sim_step!(sim, t; remeasure = false)
    first(WaterLily.pressure_force(sim)) / (0.5 * sim.L * sim.U^2)
end

times = range(20, 32; step = 0.1)
force = [inline_force!(sim, t) for t in times]
plot(times, force, xlabel = "tU/L", ylabel = "In-line force coefficient", label = "Cx")
plot!(times, -sin.(times ./ 1.5), label = "driving acceleration (scaled)", linestyle = :dash)

# The force leads the velocity, which is the inertia term dominating at this ``KC``. Sweeping
# `κ` traces out the drag and inertia coefficients as functions of the Keulegan–Carpenter
# number, which is exactly the experiment the Morison coefficients come from.

#md # ## [Plain program](@id TwoD_OscillatingFlow-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`TwoD_OscillatingFlow.jl`](TwoD_OscillatingFlow.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```

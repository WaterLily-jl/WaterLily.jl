# # [A slow-started circle](@id examples-TwoD_SlowStartCircle)
#
# ![](TwoD_SlowStartCircle.gif)
#
# *Figure 1*: A circular cylinder accelerated from rest. The symmetric pair of starting
# vortices forms first, and only later breaks down into the usual alternating street.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`TwoD_SlowStartCircle.ipynb`](TwoD_SlowStartCircle.ipynb).
#-
#
# ## Introduction
#
# Every example so far started with the free stream already at full speed everywhere, which is
# an impulsive start: at ``t=0^+`` the flow is the potential solution and the vorticity is all
# concentrated in an infinitely thin sheet on the body. That is a perfectly legitimate initial
# condition, but it is not the one a towing tank or a wind tunnel gives you, and the early
# transient is not physical.
#
# The fix is to accelerate the flow from rest. In WaterLily the second argument of
# [`Simulation`](@ref) is the boundary condition on velocity, and it accepts a **function**
# `uBC(i, x, t)` giving the ``i``-th velocity component at position `x` and time `t`, not just
# a constant tuple. Changing the inflow in time is then a one-line change.
#
# Two things to know about using a function there:
#
#  - the velocity *scale* `U` can no longer be inferred from the boundary condition, so you
#    must pass it explicitly;
#  - the function must be type stable, returning the simulation's element type `T` for every
#    component. WaterLily checks this when the `Simulation` is constructed and throws if it is
#    violated, which is why `Ut` below is written with an explicit `convert` and
#    `zero(T)` rather than `0`.
#
# ## Commented program
#
#md # The full program, without comments, can be found in the next
#md # [section](@ref TwoD_SlowStartCircle-plain-program).

using WaterLily, Plots

# ### A time-varying inflow
#
# The profile below ramps the velocity up linearly at rate `a₀` and then smoothly levels it
# off at ``u = U`` using a `tanh` blend: a constant-acceleration start followed by a
# constant-velocity tow, which is the classic experimental protocol. Only the streamwise
# component is non-zero; every other component returns `zero(T)`.
#
# The same mechanism covers any inflow you can write down: a gust, a shear profile (`uBC`
# receives the position `x`), or an oscillating free stream.

function circle(m, n; a₀ = 0.5, Re = 250, U = 1, mem = Array, T = Float32)
    r = n / 8
    body = AutoBody((x, t) -> √sum(abs2, x .- (m / 4, n / 2)) - r)
    Ut(i, x, t::T) where {T} = i == 1 ?
        convert(T, a₀ * t / r + (1 + tanh(31.4 * (t / r - 1 / a₀))) / 2 * (1 - a₀ * t / r)) :
        zero(T)
    Simulation((m, n), Ut, r; U, ν = U * r / Re, body, mem, T)
end

sim = circle(2^8, 2^7);

# Note that the body sits at a quarter of the domain length rather than in the middle: the
# starting vortices travel a long way downstream, and a short wake would let them leave
# through the exit boundary while they are still interesting.
#
# Let us look at the inflow history itself before running anything. The velocity reaches `U`
# at ``t a_0/r \approx 1/a_0``, i.e. around two convective times here.

ts = range(0, 20; step = 0.1)
plot(ts, [sim.flow.uBC(1, (0.0f0, 0.0f0), Float32(t * sim.L / sim.U)) for t in ts],
    xlabel = "tU/L", ylabel = "u/U", legend = false, title = "Inflow history")

# ### Running it
#
# Because the boundary condition is time dependent the flow is never steady in the mean, and
# the vorticity scale changes a lot between the start and the fully developed wake, hence the
# wider colour limits.

R = inside(sim.flow.p)
anim = @animate for tᵢ in range(0, 10; step = 0.1)
    sim_step!(sim, tᵢ; remeasure = false)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
    flood(sim.flow.σ[R]; clims = (-8, 8), cfill = :seismic, legend = false, border = :none)
    body_plot!(sim)
end
gif(anim, "TwoD_SlowStartCircle.gif", fps = 24)
nothing #hide

# The early wake is a symmetric pair of counter-rotating vortices growing behind the body,
# the "starting vortex" of classical aerodynamics. That state is unstable, and after a few
# convective times it buckles into the familiar alternating street. Increasing `a₀` makes the
# start more abrupt and the transition earlier; taking `a₀` small enough recovers the
# quasi-steady limit.

#md # ## [Plain program](@id TwoD_SlowStartCircle-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`TwoD_SlowStartCircle.jl`](TwoD_SlowStartCircle.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```

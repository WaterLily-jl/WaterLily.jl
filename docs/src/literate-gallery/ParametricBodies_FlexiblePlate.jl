# # [A flexible plate with ParametricBodies](@id gallery-ParametricBodies_FlexiblePlate)
#
# ![](ParametricBodies_FlexiblePlate.gif)
#
# *Figure 1*: A thin plate, defined as a NURBS curve, flexing in a cross-flow at ``Re=250``. The
# control points of the spline are drawn on top of the vorticity field.
#
#-
#md # !!! tip
#md #     This program is also available as a Jupyter notebook:
#md #     [`ParametricBodies_FlexiblePlate.ipynb`](ParametricBodies_FlexiblePlate.ipynb).
#-
#
# ## Packages
#
# - [**ParametricBodies.jl**](https://github.com/WaterLily-jl/ParametricBodies.jl) -- bodies defined
#   by parametric curves and NURBS, including ones that deform while the simulation runs.
# - [**Plots.jl**](https://github.com/JuliaPlots/Plots.jl) -- the flood plot, and the curve recipe
#   that ParametricBodies provides for drawing the spline and its control points.
#
# ParametricBodies is in the General registry, so
#
# ```julia
# pkg> add WaterLily ParametricBodies StaticArrays Plots
# ```
#
# is all it takes to run this program. Like the rest of the
# [gallery](index.md), it is *not* executed when these docs are built -- the figure
# above comes from running it separately.
#
# ## Introduction
#
# Every body in the [Examples](@ref Examples) is an [`AutoBody`](@ref): a signed distance function,
# optionally composed with a coordinate map. That covers a great deal, but it runs out when the
# shape itself is the unknown -- a membrane, a swimming fin, a structural solver's deformed mesh.
# Writing the distance to such a surface in closed form is usually not possible, and
# differentiating a map that no longer exists is certainly not.
#
# [ParametricBodies.jl](https://github.com/WaterLily-jl/ParametricBodies.jl) fills that gap. It
# defines a `ParametricBody <: WaterLily.AbstractBody` from a *parametric* description of the
# surface -- a curve ``c(s,t)`` rather than a distance field -- and supplies the `measure` that
# WaterLily needs by locating the nearest point on the curve. NURBS curves get a specialised
# locator, and a `DynamicNurbsBody` additionally tracks the velocity implied by moving control
# points, which is exactly what an immersed boundary needs from a deforming surface.
#
# This program prescribes the motion of the control points to keep it self-contained. In a real
# fluid-structure problem, those control points would come from a structural solver each step --
# that is the interface WaterLilyPreCICE.jl uses to couple WaterLily to an external code.
#
# ## Commented program
#
#md # The full program, without comments, can be found in the next
#md # [section](@ref ParametricBodies_FlexiblePlate-plain-program).

using WaterLily, ParametricBodies, StaticArrays, Plots

# ### A plate made of a spline
#
# The plate is a quadratic NURBS with three control points, written as a ``2 \times 3`` matrix of
# positions, with the weights and the knot vector that go with it. `boundary=false` says the curve
# is a thin surface rather than the outline of a closed body, so it needs a thickness: `thk` is
# given in cells, and a little over one cell is the minimum that stays watertight.

function flexible_plate(; L = 2^5, Re = 250, U = 1, mem = Array)
    ## three control points, in a 2 x 3 matrix of positions
    cps = SA[-1 0 1; 0.5 0.25 0] * L .+ [2L, 3L]
    weights, knots = SA[1.0, 1.0, 1.0], SA[0, 0, 0, 1, 1, 1.0]
    ## a thin, open curve: not the boundary of a closed body, so give it a thickness
    body = DynamicNurbsBody(NurbsCurve(cps, knots, weights); thk = √2 / 2 + 1, boundary = false)
    Simulation((8L, 6L), (U, 0), L; U, ν = U * L / Re, body, T = Float64, mem)
end

sim = flexible_plate();

# ### Driving the deformation
#
# `update!(body, new_control_points, Δt)` moves the curve and, from the displacement and the time
# step, works out the surface velocity that the immersed boundary method needs. Because the body
# changes *within* a time step rather than between frames, the inner loop here steps the flow by
# hand with `mom_step!` instead of calling `sim_step!`: update the control points, re-`measure!`
# the body onto the grid, then advance one step.

function march!(sim, tᵢ)
    t = sum(sim.flow.Δt[1:(end - 1)])
    while t < tᵢ * sim.L / sim.U
        ## the middle control point sweeps up and down; a structural solver would supply these
        new_pnts = (SA[-1 0 1; 0.5 0.25+0.5sin(π / 4 * t / sim.L) 0] .+ [2, 3]) * sim.L
        sim.body = update!(sim.body, new_pnts, sim.flow.Δt[end])
        measure!(sim, t)
        mom_step!(sim.flow, sim.pois)
        t += sim.flow.Δt[end]
    end
end

# ### Running it
#
# The plotting is the familiar flood plot, plus a recipe that ParametricBodies provides for drawing
# the curve itself. `add_cp=true` marks the control points, which is the quickest way to see that
# the spline is doing what you intended.

R = inside(sim.flow.p)
anim = @animate for tᵢ in range(0, 10; step = 0.1)
    march!(sim, tᵢ)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
    flood(sim.flow.σ[R]; clims = (-10, 10), cfill = :seismic, legend = false, border = :none)
    plot!(sim.body.curve; shift = (1.0, 1.0), add_cp = true)
end
gif(anim, "ParametricBodies_FlexiblePlate.gif", fps = 24)

# The figure at the top of the page is the result. The plate sheds from both tips as it sweeps, and
# the wake carries the signature of the motion rather than of the geometry alone -- the thing that
# makes flapping and swimming problems interesting, and the reason a deforming body is worth the
# extra machinery.
#
# ### Where this leads
#
# Replace the prescribed `new_pnts` with the output of a structural solver and this becomes a
# fluid-structure interaction program. That is what
# [WaterLilyPreCICE.jl](https://github.com/WaterLily-jl/WaterLilyPreCICE.jl) does through preCICE,
# and the body interface it drives is the one used above.

#md # ## [Plain program](@id ParametricBodies_FlexiblePlate-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`ParametricBodies_FlexiblePlate.jl`](ParametricBodies_FlexiblePlate.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```

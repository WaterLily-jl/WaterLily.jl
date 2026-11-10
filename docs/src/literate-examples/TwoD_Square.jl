# # [A square from four planes](@id examples-TwoD_Square)
#
# ![](TwoD_Square.gif)
#
# *Figure 1*: Vorticity behind a square cylinder at ``Re=250``. Compared to the circle, the
# separation points are pinned to the upstream corners and the wake is wider.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`TwoD_Square.ipynb`](TwoD_Square.ipynb).
#-
#
# ## Introduction
#
# The [previous example](@ref examples-TwoD_Circle) used a single distance function for a
# single smooth body. Real geometries are rarely that convenient, but they can almost always
# be *assembled* out of simple pieces, and so can their distance functions: this is
# constructive solid geometry (CSG).
#
# WaterLily defines the set operations ``\cup`` (union, `+`), ``\cap`` (intersection) and
# `-` (difference) on any two `AbstractBody`s. They build a lazy `SetBody` that applies
# `min`, `max` or negation to the measurements of its two halves, which means a combined body
# costs nothing to construct and is still differentiable, GPU-compatible, and can itself be
# combined further.
#
# Here we use that to build a square as the intersection of four half-spaces, and we meet the
# second argument of [`AutoBody`](@ref): a coordinate `map` that transforms the point before
# the distance function sees it. Defining a body in its own natural frame and *mapping* it
# into place is almost always easier than writing the distance function for the body where it
# sits, and it is how moving bodies are handled too (see the
# [hovering plate](@ref examples-TwoD_Hover)).
#
# ## Commented program
#
#md # The full program, without comments, can be found in the next
#md # [section](@ref TwoD_Square-plain-program).
#
# `StaticArrays` provides the `SA[...]` syntax for small stack-allocated vectors. Use it for
# anything inside a distance or map function: a regular `[...]` array would allocate on every
# call, and will not work on a GPU.

using WaterLily, StaticArrays, Plots

# ### A square as an intersection of planes
#
# The signed distance to a plane through `center` with unit `normal` is just the projection
# of the offset onto the normal, positive on the side the normal points to. The region where
# all four of our planes return a negative distance is the inside of the square, and the
# distance function of an intersection is the `max` of the parts, which is what `∩` builds.
#
# The `map` subtracts a fixed offset, placing the square's own origin at ``(2L, 2L)`` in the
# domain. Note it is applied to each plane separately: set operations compose bodies, so each
# half can carry its own mapping.

function make_sim(; L = 2^5, U = 1, Re = 250, mem = Array, T = Float32)
    ## signed distance to a plane
    function plane(x, t, center, normal)
        normal = normal / √sum(abs2, normal)
        sum((x .- center) .* normal)
    end
    ## place the body's origin at the centre of the near field
    map(x, t) = x - SA[2L, 2L]
    ## the square is the intersection of four half-spaces
    body = AutoBody((x, t) -> plane(x, t, SA[-L / 2, 0], SA[-1, 0]), map) ∩
        AutoBody((x, t) -> plane(x, t, SA[0, L / 2], SA[0, 1]), map) ∩
        AutoBody((x, t) -> plane(x, t, SA[L / 2, 0], SA[1, 0]), map) ∩
        AutoBody((x, t) -> plane(x, t, SA[0, -L / 2], SA[0, -1]), map)
    Simulation((8L, 4L), (U, 0), L; U, ν = U * L / Re, body, mem, T)
end

sim = make_sim();

# ### Checking the geometry
#
# It is worth *looking* at a constructed distance function before running anything with it:
# a wrong sign or a missing normalisation shows up immediately. `measure_sdf!` fills an array
# with ``d(x)`` for the whole domain; the zero contour is the surface.

R = inside(sim.flow.p)
measure_sdf!(sim.flow.σ, sim.body)
flood(sim.flow.σ[R]; clims = (-2sim.L, 2sim.L), levels = 20, legend = false)
body_plot!(sim)

# The distance field is exact outside the square and inside the four quadrant corners, and
# only approximate in the corner regions where `max` picks the wrong plane. That is harmless here,
# since the immersion kernel only uses ``d`` within a couple of cells of the surface.
#
# ### Running it
#
# From here on nothing differs from the circle: the body is just a body.

anim = @animate for tᵢ in range(0, 10; step = 0.1)
    sim_step!(sim, tᵢ; remeasure = false)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
    flood(sim.flow.σ[R]; clims = (-10, 10), cfill = :seismic, legend = false, border = :none)
    body_plot!(sim)
end
gif(anim, "TwoD_Square.gif", fps = 24)
nothing #hide

# The drag coefficient of a square is roughly twice that of a circle at the same Reynolds
# number, for the same reason the wake is wider: the separation points cannot move.

WaterLily.pressure_force(sim) ./ (0.5 * sim.L * sim.U^2)

# Try `-` instead of `∩` to cut one shape out of another, or `+` to glue shapes together;
# the [next example](@ref examples-TwoD_MultipleBodies) uses the latter to immerse a whole
# cloud of bodies at once.

#md # ## [Plain program](@id TwoD_Square-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`TwoD_Square.jl`](TwoD_Square.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```

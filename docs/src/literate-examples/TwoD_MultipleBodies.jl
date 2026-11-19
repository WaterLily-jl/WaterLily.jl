# # [Many bodies at once](@id examples-TwoD_MultipleBodies)
#
# ![](TwoD_MultipleBodies.gif)
#
# *Figure 1*: Flow through a random cloud of six circular cylinders at ``Re=550``.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`TwoD_MultipleBodies.ipynb`](TwoD_MultipleBodies.ipynb).
#-
#
# ## Introduction
#
# The [square](@ref examples-TwoD_Square) was one body built from four pieces. The same
# machinery immerses many *separate* bodies: the union of a collection of distance functions
# is their pointwise minimum, and `∪` (equivalently `+`) is exactly that. Because the
# combination is a lazy binary tree of `SetBody`s, `sum` over a collection of bodies just
# works, and there is no practical limit to how many you add: the cost is one distance
# evaluation per body per near-surface cell.
#
# That raises a question the single-body examples did not: when the force on *one* of the
# bodies is what you want, how do you get it out of a flow field that only knows about the
# combined geometry? This example shows both the aggregate force and the per-body
# decomposition, which is useful for things like the load on an individual tube in a bundle,
# or on a single blade in a row.
#
# ## Commented program
#
#md # The full program, without comments, can be found in the next
#md # [section](@ref TwoD_MultipleBodies-plain-program).

using WaterLily, StaticArrays, Plots
using Random #hide
Random.seed!(1234) #hide

# ### Summing bodies
#
# Six circles with random centres and radii, summed into a single body. `sum` folds the
# collection with `+`, which is `∪`, so the resulting distance function is the minimum over
# all six, that is the distance to the *nearest* body, which is what the immersion kernel needs.

function cloud(n, m; Re = 550, U = 1, mem = Array, T = Float32)
    R, x₀ = m ÷ 18, m ÷ 2
    body = sum(1:6) do _
        center, radius = 10rand(SVector{2,T}) .- 5, (3rand(T) + 3) / 4
        AutoBody((x, t) -> √sum(abs2, x .- x₀ .- center * R) - radius * R)
    end
    Simulation((n, m), (U, 0), R; U, ν = U * R / Re, body, mem, T)
end

sim = cloud(3 * 2^7, 2^8);

# The combined distance field shows all six bodies at once. Note how `min` keeps the field a
# valid distance function everywhere except in the "ridges" midway between two bodies, which
# again sit far enough from every surface not to matter.

R = inside(sim.flow.p)
measure_sdf!(sim.flow.σ, sim.body)
flood(sim.flow.σ[R]; clims = (-2sim.L, 2sim.L), levels = 20, legend = false)
body_plot!(sim)

# ### Running it
#
# The wake of a cloud like this is much more chaotic than a single cylinder's: each body sheds
# into the one behind it, so the shedding never settles into a clean periodic state.

anim = @animate for tᵢ in range(0, 10; step = 0.1)
    sim_step!(sim, tᵢ; remeasure = false)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
    flood(sim.flow.σ[R]; clims = (-5, 5), cfill = :seismic, legend = false, border = :none)
    body_plot!(sim)
end
gif(anim, "TwoD_MultipleBodies.gif", fps = 24)
nothing #hide

# ### Net force and per-body force
#
# `WaterLily.pressure_force(sim)` integrates over every surface in `sim.body`, so it gives the
# *net* force on the cloud.

f_total = WaterLily.pressure_force(sim)

# To get the force on each body separately, call the same metric on each leaf of the body
# tree. `pressure_force(flow, body)` takes the body as a separate argument precisely so that
# it can be handed a sub-body, and a union is a `SetBody{typeof(min)}` whose `.a` and `.b`
# fields are its two halves. Recursing down the tree and `hcat`-ing the results gives one
# force column per body.

each_force(flow, body) = WaterLily.pressure_force(flow, body)
each_force(flow, body::WaterLily.SetBody{typeof(min)}) =
    mapreduce(b -> each_force(flow, b), hcat, (body.a, body.b))

f_each = each_force(sim.flow, sim.body)

# The columns sum back to the net force, as they must, because the surface integral is additive:

vec(sum(f_each, dims = 2)) ≈ f_total

# A word of warning: this decomposition is only meaningful where the bodies are far enough
# apart that their immersion kernels do not overlap. Two surfaces within a couple of cells of
# each other share the same blurred interface, and the split between them becomes arbitrary.

#md # ## [Plain program](@id TwoD_MultipleBodies-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`TwoD_MultipleBodies.jl`](TwoD_MultipleBodies.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```

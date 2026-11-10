# # [Flow over a sphere](@id examples-ThreeD_Sphere)
#
# ![](ThreeD_Sphere.gif)
#
# *Figure 1*: Vorticity magnitude ``|\omega| L/U`` on the centre-plane of the flow over a sphere at
# ``Re=250``, rendered with Makie through WaterLily's `viz!`.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`ThreeD_Sphere.ipynb`](ThreeD_Sphere.ipynb).
#-
#
# ## Introduction
#
# The last example was 3D but had no body; the [first example](@ref examples-TwoD_Circle) had a
# body but was 2D. This one puts the two together: the flow over a sphere at ``Re=250``, which is
# the 3D counterpart of the circle and the most-measured bluff-body flow there is. Just past
# ``Re \approx 210`` the steady axisymmetric wake loses its symmetry and begins to shed hairpin
# vortices, so there is something to look at.
#
# The point of this example is the part that is *not* about geometry:
#
#  - measuring forces properly, splitting the drag into its **pressure** and **viscous** parts.
#    In 2D at ``Re=250`` the viscous contribution is small enough to ignore; on a sphere it is
#    not.
#  - checking that the answer means anything, by **refining the grid** and seeing whether the
#    drag coefficient moves.
#  - what changes when you move the whole thing to a **GPU**, which for 3D is less a luxury than
#    the only practical option.
#
# ## Commented program
#
#md # The full program, without comments, can be found in the next
#md # [section](@ref ThreeD_Sphere-plain-program).

using WaterLily, CairoMakie

# ### The geometry
#
# A sphere is the same one-line distance function as a circle: ``|x - c| - r`` does not care how
# many components `x` has. This is the whole of the 2D-to-3D change in the setup.

function sphere(L = 2^4; Re = 250, U = 1, mem = Array, T = Float32)
    center, radius = (L, L, L), L / 2
    body = AutoBody((x, t) -> √sum(abs2, x .- center) - radius)
    Simulation((3L, 2L, 2L), (U, 0, 0), L; U, ν = U * L / Re, body, mem, T)
end

# ### Looking at a 3D field with Makie
#
# A flood plot of a 3D array is not a thing, so visualising this needs more than `Plots`. WaterLily
# has a Makie extension for exactly this: load any Makie backend and you get `viz!`, which runs the
# time loop *and* renders it, with the body drawn in.
#
# `viz!` takes the field to draw as a function `f(arr, sim)` that fills a CPU array -- the same
# scratch-array-and-`@inside` pattern as every metric in these examples, plus a `copyto!` so that
# the data comes back from the GPU when there is one.

function ω!(arr, sim)
    a = sim.flow.σ
    @inside a[I] = WaterLily.ω_mag(I, sim.flow.u) * sim.L / sim.U
    copyto!(arr, a[inside(a)])
end

# ### Measuring the drag
#
# `WaterLily.pressure_force` integrates ``p\,n`` over the immersed surface and
# `WaterLily.viscous_force` integrates the viscous traction; `WaterLily.total_force` is their sum.
# For a sphere the reference area is ``\pi r^2``, so the drag coefficient is
# ``C_d = F_x / (\tfrac12 \rho U^2 \pi r^2)``, with ``\rho = 1``.
#
# Keeping the two parts separate is worth the extra line: their ratio tells you whether the flow
# is pressure- or friction-dominated, and a viscous force that refuses to converge under grid
# refinement is the classic symptom of an under-resolved boundary layer.

function drag_coefficients(sim)
    area = π * (sim.L / 2)^2
    scale = 0.5 * sim.U^2 * area
    cp = first(WaterLily.pressure_force(sim)) / scale
    cf = first(WaterLily.viscous_force(sim)) / scale
    cp, cf
end

# ### Running it
#
# First a plain time loop for the forces, so the drag history starts from rest. A
# ``48 \times 32 \times 32`` grid puts only 16 cells across the sphere, which is coarse -- it is
# chosen so that this page builds in a reasonable time.

sim = sphere(2^4)
times = range(0, 10; step = 0.1)
cd_history = Tuple{Float64,Float64}[]
for tᵢ in times
    sim_step!(sim, tᵢ; remeasure = false)
    push!(cd_history, drag_coefficients(sim))
end

# The pressure and viscous contributions to the drag, and their sum. This example uses Makie for
# its plots as well as its rendering: `Plots` and `Makie` both export `plot`, so loading both into
# one session makes that name ambiguous, and picking one per program avoids the problem.

fig = Figure(size = (700, 350))
ax = Axis(fig[1, 1], xlabel = "tU/L", ylabel = "Cd")
lines!(ax, times, first.(cd_history), label = "pressure")
lines!(ax, times, last.(cd_history), label = "viscous")
lines!(ax, times, first.(cd_history) .+ last.(cd_history), label = "total")
axislegend(ax)
fig

# ### Rendering the wake
#
# Now hand the next few convective times to `viz!`, which steps the simulation and renders each
# frame. `d=2` asks for a plane rather than a volume and `cut=(0,0,L)` picks the mid-plane through
# the sphere; `video` names the file to record, and `verbose=false` keeps the per-frame progress
# out of this page.

viz!(sim; f = ω!, duration = 5, step = 0.1, remeasure = false, verbose = false,
    d = 2, cut = (0, 0, 2^4), video = "ThreeD_Sphere.gif", framerate = 24,
    colormap = :inferno, clims = (0, 10), levels = 20, hidedecorations = true)
nothing #hide

# The animation at the top of the page is the result. The body is drawn by `viz!` itself -- it has
# the simulation, so it knows the geometry and does not need to be told.
#
# The *really* nice rendering is the one this page cannot build: drop `d` and `cut` and `viz!`
# volume-renders the vorticity in 3D, which is worth doing once interactively to see what the
# wake of a sphere actually looks like.
#
# ```julia
# using GLMakie                # a GPU backend, for volume rendering and an interactive window
# viz!(sim; f = ω!, duration = 5, colormap = :inferno, clims = (0, 10))
# ```
#
# That needs an OpenGL context, which a documentation build on CI does not have, so the plane
# above is what gets rendered here. Both calls are otherwise identical.

# ### Does the answer mean anything?
#
# A single number from a single grid is not a result. Running the same case at half the resolution
# and comparing is the cheapest possible convergence check, and it is the one you should always do
# before believing a force.

mean_drag(history, times; from = 5) =
    (idx = findall(≥(from), times); sum(sum, history[idx]) / length(idx))

fine = mean_drag(cd_history, times)

# and the same case on a grid half as fine in every direction:

function run_drag(L; duration = 10)
    sim = sphere(L)
    ts = range(0, duration; step = 0.1)
    history = [(sim_step!(sim, t; remeasure = false); drag_coefficients(sim)) for t in ts]
    mean_drag(history, ts)
end

coarse = run_drag(2^3)

# The experimental drag coefficient of a sphere at ``Re=250`` is about ``0.7``. Both of these grids
# are too coarse to get there, and the gap between them tells you so, and that is the useful
# information. The resolution at which the two agree to within a percent or so is the resolution
# you should be running, and for this flow it is nearer 32 cells per diameter than 16.
#
# ### Moving it to a GPU
#
# 3D is where the grid cost bites: going from 16 to 32 cells across the sphere multiplies the work
# by eight. WaterLily uses
# [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), so the same code runs
# on a GPU with one keyword. Every example on these pages takes a `mem` argument for exactly this
# reason.
#
# ```julia
# using CUDA               # or AMDGPU, for a ROCm device
# sim = sphere(2^6; mem = CuArray)
# sim_step!(sim, 10; remeasure = false)
# ```
#
# That is a ``192 \times 128 \times 128`` grid: over three million cells, and a simulation that is
# still comfortable on a single GPU. Two things to keep in mind:
#
#  - the flow arrays now live in device memory. The metrics (`pressure_force` and friends) return
#    host arrays and work unchanged, but anything you hand to a plotting library needs an explicit
#    `|> Array` or `copyto!` to come back to the CPU first.
#  - keep everything inside a distance function, a `map`, a `uBC` or a `g` allocation-free and type
#    stable: `SA[...]` vectors rather than `[...]`, and `zero(T)` rather than `0`. Those functions
#    are compiled into GPU kernels, and that is the whole of what the GPU requires of you.

#md # ## [Plain program](@id ThreeD_Sphere-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`ThreeD_Sphere.jl`](ThreeD_Sphere.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```

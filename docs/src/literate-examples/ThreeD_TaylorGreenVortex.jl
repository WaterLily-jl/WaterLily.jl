# # [Taylor-Green vortex](@id examples-ThreeD_TaylorGreenVortex)
#
# ![](ThreeD_TaylorGreenVortex.gif)
#
# *Figure 1*: Decay of the Taylor-Green vortex at ``Re=1600``, shown as ``\log_{10}(-\lambda_2)``
# on the mid-plane ``z = L/2``. The smooth initial vortex array rolls up, breaks down into
# small-scale structures, and then decays.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`ThreeD_TaylorGreenVortex.ipynb`](ThreeD_TaylorGreenVortex.ipynb).
#-
#
# ## Introduction
#
# This is the first three-dimensional example, and the first with no body in it at all. The
# Taylor-Green vortex is the standard test case for the accuracy of a transitional flow solver:
# a smooth, closed-form initial velocity field in a cube, left to evolve with no forcing and no
# boundaries. It stretches and tilts itself into progressively finer structures, the dissipation
# rises to a sharp peak as the small scales appear, and then everything decays. Reference data
# for the integrated quantities exists to many digits, so it is a genuine check and not just a
# pretty picture.
#
# Two new things appear:
#
#  - the **initial condition** `u0`, a function `u0(i, x)` giving the ``i``-th velocity
#    component at position `x`. Until now every example started from a uniform stream, which is
#    what WaterLily infers from the boundary condition when `u0` is not given.
#  - going from 2D to 3D, which (as you will see) changes the code almost not at all. The
#    arrays grow a dimension, [`@inside`](@ref) loops over three indices instead of two, and
#    vorticity becomes a vector rather than a scalar. Everything else is identical.
#
# ## Commented program
#
#md # The full program, without comments, can be found in the next
#md # [section](@ref ThreeD_TaylorGreenVortex-plain-program).

using WaterLily, Plots

# ### The initial condition
#
# The Taylor-Green field on a cube of side ``L`` with wavenumber ``\kappa = \pi/L`` is
#
# ```math
# u_x = -U \sin(\kappa x)\cos(\kappa y)\cos(\kappa z), \quad
# u_y = \ \ U \cos(\kappa x)\sin(\kappa y)\cos(\kappa z), \quad
# u_z = 0 .
# ```
#
# Like `uBC` and `g`, the function must be type stable, which is why `U` and `κ` are converted to
# the simulation's element type `T` up front.
#
# Notice that the normal velocity vanishes on all six faces of the cube: ``u_x = 0`` at
# ``x = 0, L`` and so on. The free-slip, impermeable box WaterLily gives us with a `(0,0,0)`
# boundary condition is therefore exactly equivalent, by symmetry, to the periodic problem the
# reference data is computed for, with no periodic directions needed.

function TGV(L; Re = 1600, U = 1, T = Float32, mem = Array)
    κ, U = T(π / L), T(U)
    function u0(i, xyz)
        x, y, z = @. xyz * κ
        i == 1 && return -U * sin(x) * cos(y) * cos(z)
        i == 2 && return U * cos(x) * sin(y) * cos(z)
        return zero(U)
    end
    Simulation((L, L, L), (0, 0, 0), L; U, u0, ν = U * L / Re, T, mem)
end

sim = TGV(2^5);

# A cube of ``32^3`` cells at ``Re=1600`` is under-resolved (the reference spectral solutions
# use ``512^3``), but it reproduces the qualitative behaviour and the position of the dissipation
# peak, and it keeps this page buildable. Increase to `2^6` or `2^7` and, since a 3D simulation
# gets expensive quickly, move it to a GPU with `mem=CuArray`.
#
# ### Looking at a 3D field
#
# The `λ₂` criterion is the standard way to pick vortex cores out of a 3D velocity field: it is
# the intermediate eigenvalue of the symmetric part of the velocity-gradient tensor squared, and
# it is strongly negative inside a vortex. Plotting ``\log_{10}(-\lambda_2)`` compresses its huge
# dynamic range into something a colour map can show.
#
# For a figure in a web page we take a slice. `size(sim.flow.p, 3) ÷ 2` is the mid-plane, and
# everything else is the 2D plotting we have used all along.

function λ₂_slice(sim)
    @inside sim.flow.σ[I] = log10(max(1.0f-6, -WaterLily.λ₂(I, sim.flow.u) * sim.L / sim.U))
    k = size(sim.flow.p, 3) ÷ 2
    sim.flow.σ[inside(sim.flow.p)][:, :, k]
end

# ### Integrated quantities
#
# The numbers everyone compares are integrals over the whole cube. WaterLily's `ke` metric gives
# the kinetic energy per cell, so its mean over the interior is ``E(t)``, whose decay rate is the
# dissipation ``\varepsilon = -dE/dt``; the enstrophy ``\tfrac12\langle|\omega|^2\rangle`` is the
# other standard diagnostic, and for an incompressible flow in a box the two are directly related.
#
# Both reuse `sim.flow.σ` as scratch space, which is the normal way to compute a cell-wise
# quantity in WaterLily without allocating.

function energy_and_enstrophy(sim)
    R = inside(sim.flow.p)
    @inside sim.flow.σ[I] = WaterLily.ke(I, sim.flow.u)
    E = sum(sim.flow.σ[R]) / length(R) / sim.U^2
    @inside sim.flow.σ[I] = WaterLily.ω_mag(I, sim.flow.u)^2
    Ω = 0.5 * sum(sim.flow.σ[R]) / length(R) * (sim.L / sim.U)^2
    E, Ω
end

# ### Running it
#
# One pass through time produces both the animation and the history; there is no reason to run a
# 3D simulation twice.

times = range(0, 10; step = 0.1)
history = Tuple{Float64,Float64}[]
anim = @animate for tᵢ in times
    sim_step!(sim, tᵢ; remeasure = false)
    push!(history, energy_and_enstrophy(sim))
    flood(λ₂_slice(sim); clims = (-3, 1), cfill = :Reds, legend = false, border = :none)
end
gif(anim, "ThreeD_TaylorGreenVortex.gif", fps = 24)
nothing #hide

p1 = plot(times, first.(history), xlabel = "tU/L", ylabel = "E/U²", legend = false,
    title = "Kinetic energy")
p2 = plot(times, last.(history), xlabel = "tU/L", ylabel = "Ω L²/U²", legend = false,
    title = "Enstrophy")
plot(p1, p2, layout = (1, 2), size = (800, 350))

# The enstrophy peaks around ``tU/L \approx 3``, which is where the vortex sheets have rolled up
# into the finest structures the grid can carry and the dissipation is largest. That is the feature the
# reference data pins down. At this resolution the peak is too low and too late; refining the grid
# moves it towards the published value, and that convergence is the actual test.

#md # ## [Plain program](@id ThreeD_TaylorGreenVortex-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`ThreeD_TaylorGreenVortex.jl`](ThreeD_TaylorGreenVortex.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```

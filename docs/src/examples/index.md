# Examples

On this page you find an overview of the WaterLily examples. Every example is a *commented
program*: the code that solves the problem, spliced with the explanation of what each piece
does and why. The figures on each page are produced by running that same code when the
documentation is built, so what you read is what was run.

The examples all follow the same structure:

- **Introduction** states the problem and what the example adds to the ones before it.
- **Commented program** is the code, with explanations.
- **Plain program** is the raw source, also downloadable as a `.jl` script or a Jupyter
  notebook from the links at the top of each page.

The simulations are deliberately small so that the documentation builds in a reasonable
time. Every example notes the knob to turn (usually the grid resolution `L` or the
`duration`) to get a better-resolved result on your own machine, and nearly all of them run
unchanged on a GPU by passing `mem=CuArray`. A much larger collection lives in the
[WaterLily-Examples](https://github.com/WaterLily-jl/WaterLily-Examples) repository.

### Example index

The examples are listed in roughly increasing order of complexity, and each one assumes the
ones before it. [Flow over a circle](TwoD_Circle.md) is the place to start: it introduces
the `Simulation` object, the `sim_step!` loop and the plotting helpers that every other
example reuses.

```@raw html
<div class="gallery-card"><div class="gallery-card-figure">
```

![](TwoD_Circle.gif)

```@raw html
</div><div class="gallery-card-body">
```

#### [Flow over a circle](TwoD_Circle.md)

The "Hello, world!" of WaterLily. Builds a circle from a signed distance function, steps the
flow forward, animates the vorticity field, and measures the pressure force on the body.

**Keywords**: `Simulation`, `AutoBody`, `sim_step!`, `sim_gif!`, `pressure_force`.

```@raw html
</div></div>
```

```@raw html
<div class="gallery-card"><div class="gallery-card-figure">
```

![](TwoD_Square.gif)

```@raw html
</div><div class="gallery-card-body">
```

#### [A square from four planes](TwoD_Square.md)

The same flow around a sharp-edged body, built by intersecting the half-spaces of four
planes. Introduces constructive solid geometry on signed distance functions and the `map`
argument that moves a body through the domain.

**Keywords**: constructive solid geometry, `∩`, coordinate `map`.

```@raw html
</div></div>
```

```@raw html
<div class="gallery-card"><div class="gallery-card-figure">
```

![](TwoD_MultipleBodies.gif)

```@raw html
</div><div class="gallery-card-body">
```

#### [Many bodies at once](TwoD_MultipleBodies.md)

Six randomly placed circles immersed in the same flow, assembled by summing `AutoBody`s.
Shows how to recover the force on the whole collection and on each body separately.

**Keywords**: `SetBody`, `∪`, body union, per-body forces.

```@raw html
</div></div>
```

```@raw html
<div class="gallery-card"><div class="gallery-card-figure">
```

![](TwoD_SlowStartCircle.gif)

```@raw html
</div><div class="gallery-card-body">
```

#### [A slow-started circle](TwoD_SlowStartCircle.md)

Instead of a constant free stream, the inflow is a function of time that accelerates the
flow from rest. Introduces time-varying boundary conditions and the starting vortex.

**Keywords**: time-varying `uBC`, impulsive start.

```@raw html
</div></div>
```

```@raw html
<div class="gallery-card"><div class="gallery-card-figure">
```

![](TwoD_OscillatingFlow.gif)

```@raw html
</div><div class="gallery-card-body">
```

#### [Oscillating flow over a circle](TwoD_OscillatingFlow.md)

A circle in a flow that reverses periodically, driven by a time-varying body force rather
than by the boundary. Also introduces periodic domain directions.

**Keywords**: body force `g`, `perdir`, periodic boundary conditions.

```@raw html
</div></div>
```

```@raw html
<div class="gallery-card"><div class="gallery-card-figure">
```

![](TwoD_Hover.gif)

```@raw html
</div><div class="gallery-card-body">
```

#### [A hovering plate](TwoD_Hover.md)

A flat plate that pitches and heaves in otherwise still fluid. Introduces moving geometry:
the body is defined once in its own frame and moved by a time-dependent `map`, which
requires remeasuring the body each step.

**Keywords**: moving bodies, `map`, `remeasure`, `ϵ`.

```@raw html
</div></div>
```

```@raw html
<div class="gallery-card"><div class="gallery-card-figure">
```

![](ThreeD_TaylorGreenVortex.gif)

```@raw html
</div><div class="gallery-card-body">
```

#### [Taylor-Green vortex](ThreeD_TaylorGreenVortex.md)

The first 3D example, and the first without a body: a closed-form initial velocity field
that decays into turbulence. Introduces initial conditions, the `λ₂` vortex criterion, and
how to look at a 3D field.

**Keywords**: 3D, initial condition `u0`, `λ₂`, kinetic energy decay.

```@raw html
</div></div>
```

```@raw html
<div class="gallery-card"><div class="gallery-card-figure">
```

![](ThreeD_Sphere.gif)

```@raw html
</div><div class="gallery-card-body">
```

#### [Flow over a sphere](ThreeD_Sphere.md)

A 3D immersed body with force measurement, rendered with Makie through WaterLily's `viz!`, and a
grid-refinement check of the drag coefficient. Closes with what changes when the same simulation
is moved onto a GPU.

**Keywords**: 3D, `pressure_force`, `viscous_force`, `viz!`, Makie, grid convergence, GPU backends.

```@raw html
</div></div>
```

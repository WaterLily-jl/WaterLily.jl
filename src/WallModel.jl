"""
    log_law(U,y,ν;κ=0.41,B=5.2)

Friction velocity `uτ` from the wall-parallel speed `U` at wall distance `y` using the two-layer
law of the wall

    u⁺ = y⁺              for y⁺ < y⁺c (viscous sublayer)
    u⁺ = log(y⁺)/κ+B     otherwise    (log layer)

where `u⁺=U/uτ`, `y⁺=y*uτ/ν` and `y⁺c≈11` is the intersection of the two branches.
The log layer is solved with Newton iterations started above the root, where the residual
is convex, so they converge monotonically. Requires `ν>0`.
"""
@fastmath @inline function log_law(U::T,y,ν;κ=0.41,B=5.2) where T
    κ,B,y,ν = T(κ),T(B),T(y),T(ν)
    y⁺ = T(11)
    for _ in 1:4 # fixed-point for the sublayer/log-layer intersection
        y⁺ = log(y⁺)/κ+B
    end
    U*y < ν*y⁺^2 && return √(ν*U/y)
    uτ = U/y⁺
    for _ in 1:6
        a = log(y*uτ/ν)/κ+B
        uτ -= (uτ*a-U)/(a+inv(κ))
    end
    return uτ
end

# Linear interpolation of the vector field `u` at `x`, as `interp`, but with the query wrapped
# in the periodic directions `perdir` instead of clamped to the domain.
@inline function interp_per(x::SVector{D,T},u::AbstractArray{T},perdir) where {D,T}
    sz = Base.front(size(u))
    @inline at(i) = SVector{D,T}(ntuple(Val(D)) do j
        xⱼ = x[j]-stagger(i,D,T)[j]; L = T(sz[j]-2)
        j in perdir ? mod(xⱼ,L) : clamp(xⱼ,zero(T),L)
    end)
    SVector{D,T}(ntuple(i->_interp(at(i),u,i),Val(D)))
end

# Wall-parallel velocity relative to the body, sampled a distance `h` above the wall along
# the normal through `x`, where `d,n,V` are the body measurements at `x`.
@inline function u_parallel(u,x::SVector{D,T},d,n,V,h,perdir) where {D,T}
    uₛ = interp_per(x+(h-d)*n,u,perdir)-V
    uₛ-(uₛ'*n)*n
end

# Wall-normal velocity relative to the body at the face `x` of component `i`, from the velocity
# interpolated through the two adjacent cell centres. Every component then sees the same stencil
# across a sheared wall-parallel flow. Interpolating at the face itself reads the curvature of
# that flow as a wall-normal velocity which differs between the faces of a cell, and removing
# it drains wall-parallel momentum on walls that are oblique to the grid.
@inline function u_normal(u,x::SVector{D,T},i,n,V,perdir) where {D,T}
    e = SVector{D,T}(ntuple(k->ifelse(k==i,T(0.5),zero(T)),Val(D)))
    n'*((interp_per(x-e,u,perdir)+interp_per(x+e,u,perdir))/2-V)
end

# Wall-parallel velocity for a face `x` on the solid side: the mean over the neighbours along the
# grid axes that are on the fluid side, so the net diffusive flux between the two sides is zero
# for any wall orientation. Faces without such a neighbour sample at the height `hₛ` instead.
@inline function u_extended(u,body,x::SVector{D,T},d,n,V,t,hₛ,perdir) where {D,T}
    s = zero(x); m = 0
    if d>-1
        for j in 1:D
            xⱼ = x+SVector{D,T}(ntuple(k->ifelse(k==j,sign(n[j]),zero(T)),Val(D)))
            if sdf(body,xⱼ,t)>0
                s += u_parallel(u,xⱼ,zero(T),n,V,zero(T),perdir); m += 1
            end
        end
    end
    m>0 ? s/m : u_parallel(u,x,d,n,V,hₛ,perdir)
end

# Slip target and wall-stress force on the face `Ii`
@inline function wm_face!(f,V,u⁰,u,body,Ii,t,ν,ϵ,h,hₛ,w,d₀,law,perdir)
    T = eltype(f); i = last(Ii); x = loc(Ii,T)
    R = max(ϵ+2max(w,one(T)),d₀+2w)
    d,n,Vᵦ = measure(body,x,t;fastd²=R^2)
    d^2 ≥ R^2 && return nothing
    # Slip target from the velocity at the start of the step, filled beyond the kernel so that
    # `μddn` in `BDIM!` sees no jump at its edge.
    # Fluid side: the face's own velocity minus its wall-normal part, so the blend only removes
    # that part and `u∥ = u⁰∥ + μ₀Δt r∥` without any relaxation.
    # Solid side: zero-gradient extension of the fluid velocity next to it.
    V[Ii] = d>0 ? u⁰[Ii]-n[i]*u_normal(u⁰,x,i,n,Vᵦ,perdir) : Vᵦ[i]+u_extended(u⁰,body,x,d,n,Vᵦ,t,hₛ,perdir)[i]
    # Wall stress on the fluid side, over the layer d₀ < d < d₀+2w
    (d≤d₀ || d≥d₀+2w) && return nothing
    uₚ = u_parallel(u,x,d,n,Vᵦ,h,perdir); U = √(uₚ'*uₚ)
    U>eps(T) && (f[Ii] -= law(U,h,ν)^2*kern((d-d₀-w)/w)/w*uₚ[i]/U)
    return nothing
end

"""
    wall_model!(flow, u, t; body, ϵ=1, w=1, d₀=0, h=max(ϵ,d₀)+2w, hₛ=1/2, wall_law=log_law)

User-defined function `udf` applying an algebraic wall-stress model on an immersed `body`.
The no-slip condition of the BDIM update is replaced by two pieces:

  - *Slip*: `flow.V` is overwritten with the blending target `V = V_body + u∥`, which the pressure
    projection then uses consistently, with `u∥` taken from the velocity at the start of the step.
    On the fluid side (`d>0`) `u∥` is the face's own wall-parallel velocity, so the blend only acts
    on the wall-normal component (no penetration) and the wall-parallel momentum is not relaxed:
    `u∥ = u⁰∥ + μ₀Δt r∥`, up to the `μ₁` correction. On the solid
    side (`d≤0`) `u∥` is the mean over the neighbouring faces along the grid axes that lie on the
    fluid side, or a sample at the height `hₛ` when there is none. This extends the fluid velocity
    into the body with zero net diffusive flux through the wall, whatever its orientation.
  - *Wall stress*: `u∥` is sampled at the matching height `h` and `wall_law(|u∥|,h,ν)` returns the
    friction velocity `uτ`. The stress `τ = uτ² u∥/|u∥|` is applied as the body force `-τ K(d)`,
    with `K` a unit-integral cosine kernel of half-width `w` spanning `d₀ < d < d₀+2w`.

`ϵ` must be the BDIM kernel width of the simulation, `sim.ϵ`. `wall_law(U,y,ν)` defaults to
[`log_law`](@ref) and can be any function with that signature. Use as

    sim_step!(sim; udf=WaterLily.wall_model!, body=sim.body, ϵ=sim.ϵ)

The sample points are wrapped in the periodic directions and clamped to the domain otherwise.
Inside the kernel the fluid-side faces respond to the momentum fluxes and forces at roughly the
rate `μ₀` instead of 1, which affects transients but not the mean momentum balance of a stationary
flow. The split at `d=0` is sharp: a row of faces lying exactly on the wall counts as solid.

Tested on static walls and on walls sliding along themselves (the model acts on the velocity
relative to the body). Deforming bodies and bodies moving along their normal are untested.

!!! warning "Walls oblique to the grid"
    The model is only reliable on walls aligned with the grid. On an oblique wall the near-wall
    stream passes back and forth through the staircase of faces between the fluid and solid
    sides, carrying about 30 times the momentum that the wall stress removes, and the two
    directions do not cancel exactly: the fluid gains momentum. In channels at Re_τ=550 with 16
    cells per half-height this spurious push was 20% of the driving force at 45° and 10 to 40%
    on a 1:2 slope with `quick`, depending on the position of the wall in its cell, and larger
    with `cds` on the 1:2 slope. It grows with the share of the wall stress that is applied
    within the first cell: with `w=3` it stayed within the scatter of ±5% in short runs. See
    `examples/checks/wm_oblique_budget.jl`.
"""
function wall_model!(flow::AbstractFlow{D,T},u,t;body,ϵ=1,w=1,d₀=0,h=max(ϵ,d₀)+2w,hₛ=1/2,wall_law=log_law) where {D,T}
    @assert flow.ν>0 "the wall model needs a viscosity ν>0"
    @assert h>0 && hₛ>0 "the sample heights must lie above the wall"
    @assert w>0 && d₀≥0 "the stress layer must lie on the fluid side"
    tᵦ = sum(flow.Δt) # as in `measure!(sim)`
    ν,ϵ,h,hₛ,w,d₀,perdir = flow.ν,T(ϵ),T(h),T(hₛ),T(w),T(d₀),flow.perdir
    @loop wm_face!(flow.f,flow.V,flow.u⁰,u,body,Ii,tᵦ,ν,ϵ,h,hₛ,w,d₀,wall_law,perdir) over Ii ∈ inside_u(size(flow.p))
    BC!(flow.V,zeros(SVector{D,T}),flow.exitBC,flow.perdir)
end

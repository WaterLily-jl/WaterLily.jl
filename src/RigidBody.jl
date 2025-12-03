"""
    RigidBody(sdf, center, θ) <: AbstractBody

  - `sdf(x::AbstractVector,t::Real)::Real`: signed distance function
  - `center::SVector{D}`: coordinate of the center of the body
  - `θ::Union{Real, SVector{3}}`: rotation (angle in 2D, euler angles in 3D)
  - `velocity::SVector{D}=zero(center)`: linear velocity of the center
  - `pivot::SVector{D}=zero(center)`: offset of the pivot point compared to center
  - `ω::Union{Real, SVector{3}}=zero(θ)`: angular velocity (scalar in 2D, vector in 3D)

Implicitly define a geometry by its `sdf` and rigid body motion parameters.
Note: the `sdf` is defined in the body's local frame, i.e. before rotation and translation.

RigodyBody motion has to be computed externally via a set od ODEs and updated in the
simulation loop:
```julia
using WaterLily,StaticArrays
body = RigidBody((x,t)->sqrt(sum(abs2,x))-4,SA{Float32}[16,16],0.f0;ω=0.1f0)
sim = Simulation((32,32),(1,0),8;body)
for n in 1:10
    # update body motion (example: constant angular velocity)
    θ_new += sim.body.θ + sim.body.ω*sim.flow.Δt[end]
    sim.body = update!(sim.body; θ=θ_new)
    # remeasure and step
    step!(sim;remeasure=true)
end
```

"""
struct RigidBody{D,T,F<:Function,A<:AbstractVector{T},R} <: AbstractBody
    sdf :: F            # signed distance function
    center :: A         # center of linear motion
    velocity :: A       # linear velocity of the center
    pivot    :: A       # offset of the pivot point compared to center
    θ :: R              # rotation (angle in 2D, euler angles in 3D)
    ω :: R              # angular velocity (scalar in 2D, vector in 3D)
    function RigidBody(sdf,center,θ,velocity,pivot,ω)
        T,D = eltype(center),length(center)
        new{D,T,typeof(sdf),typeof(center),typeof(θ)}(sdf,center,velocity,pivot,θ,ω)
    end
end
RigidBody(sdf,center,θ;velocity=zero(center),ω=zero(θ),pivot=zero(center)) = RigidBody(sdf,center,θ,velocity,pivot,ω)
function WaterLily.sdf(body::RigidBody{D,T},x,t=0;kwargs...) where {D,T}
    R = rotation(body.θ) # compute rotation matrix
    return body.sdf(R*(x-body.center-body.pivot)+body.pivot,t)
end
# 2D rotation using scalar angle
rotation(θ::T) where T = SA{T}[cos(θ) sin(θ); -sin(θ) cos(θ)]
# 3D rotation using Euler angles
rotation(θ::SVector{3,T}) where T = SA{T}[cos(θ[1])*cos(θ[2]) cos(θ[1])*sin(θ[2])*sin(θ[3])-sin(θ[1])*cos(θ[3]) cos(θ[1])*sin(θ[2])*cos(θ[3])+sin(θ[1])*sin(θ[3]);
                                          sin(θ[1])*cos(θ[2]) sin(θ[1])*sin(θ[2])*sin(θ[3])+cos(θ[1])*cos(θ[3]) sin(θ[1])*sin(θ[2])*cos(θ[3])-cos(θ[1])*sin(θ[3]);
                                               -sin(θ[2])                         cos(θ[2])*sin(θ[3])                               cos(θ[2])*cos(θ[3])]
# cross product and new specialized case in 2D scalar angular velocity
import WaterLily: ×
×(a::Number,b::SVector{2,T}) where T = a*SA[-b[2],b[1]]
function WaterLily.measure(body::RigidBody{D,T},x,t;fastd²=Inf) where {D,T}
    # eval d=f(x,t), and n̂ = ∇f
    d = WaterLily.sdf(body,x,t)
    d^2>fastd² && return (d,zero(x),zero(x)) # skip n,V
    n = ForwardDiff.gradient(x->WaterLily.sdf(body,x,t), x)
    any(isnan.(n)) && return (d,zero(x),zero(x))

    # correct general implicit fnc f(x₀)=0 to be a pseudo-sdf
    #   f(x) = f(x₀)+d|∇f|+O(d²) ∴  d ≈ f(x)/|∇f|
    m = √sum(abs2,n); d /= m; n /= m

    # The rigid body velocity is given by the rigid body motion
    # v = v + ω×(x-c)
    v = body.velocity + body.ω×(x-body.center-body.pivot)
    return (d,n,v)
end

import WaterLily: update!
"""
    update!(body::RigidBody; kwargs...)

"""
function update!(body::RigidBody; sdf=body.sdf, center=body.center,
                 velocity=body.velocity, pivot=body.pivot, θ=body.θ, ω=body.ω)
    return RigidBody(sdf,center,θ,velocity,pivot,ω)
end

"""
    Store(sim::AbstractSimulation)

Store the current state of a simulation to allow reverting back to it later.
This is useful for iterative schemes where you may want to try a step and revert if it fails.
"""
mutable struct Store
    uˢ:: AbstractArray
    pˢ:: AbstractArray
    b :: AbstractBody
    function Store(sim::AbstractSimulation)
        new(copy(sim.flow.u),copy(sim.flow.p),deepcopy(sim.body))
    end
end
function store!(s::Store,sim::AbstractSimulation)
    s.uˢ .= sim.flow.u; s.pˢ .= sim.flow.p
    s.b = deepcopy(sim.body)
end
function revert!(s::Store,sim::AbstractSimulation)
    sim.flow.u .= s.uˢ; sim.flow.p .= s.pˢ; pop!(sim.flow.Δt)
    pop!(sim.pois.n); pop!(sim.pois.n) # pop predictor and corrector
    sim.body = s.b # nice and simple
end
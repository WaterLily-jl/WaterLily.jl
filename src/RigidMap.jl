"""
    RigidMap(center, θ) <: AbstractBody

  - `center::SVector{D}`: coordinate of the center of the body
  - `θ::Union{Real, SVector{3}}`: rotation (angle in 2D, euler angles in 3D)
  - `velocity::SVector{D}=zero(center)`: linear velocity of the center
  - `pivot::SVector{D}=zero(center)`: offset of the pivot point compared to center
  - `ω::Union{Real, SVector{3}}=zero(θ)`: angular velocity (scalar in 2D, vector in 3D)

Define a `RigidMap` for any `AbstractBody` using rigid body motion parameters.

RigidMap updates are computed externally via a set of ODEs and then updated in the
simulation loop following:
```julia
using WaterLily,StaticArrays
body = AutoBody((x,t)->sqrt(sum(abs2,x))-4,RigidMap(SA{Float32}[16,16],0.f0;ω=0.1f0))
sim = Simulation((32,32),(1,0),8;body)
for n in 1:10
    # update body motion (example: constant angular velocity)
    θ = sim.body.map.θ + sim.body.map.ω*sim.flow.Δt[end]
    sim.body = setmap(sim.body; θ)
    # remeasure and step
    sim_step!(sim;remeasure=true)
end
```
"""
struct RigidMap{A<:AbstractVector,R} <: Function
    x₀ :: A   # center of translation
    θ  :: R   # rotation (angle in 2D, euler angles in 3D)
    xₚ :: A   # rotation offset
    V  :: A   # linear velocity of the center
    ω  :: R   # angular velocity (scalar in 2D, vector in 3D)
end
RigidMap(x₀::SVector,θ;xₚ=zero(x₀),V=zero(x₀),ω=zero(θ)) = RigidMap(x₀, θ, xₚ, V, ω)

# this is the function map(x,t) AND derivative(t->map(x,t),t)
(map::RigidMap)(x::SVector,t=0) = rotation(map.θ)*(x-map.x₀-map.xₚ)+map.xₚ
(map::RigidMap)(x::SVector,::ForwardDiff.Dual{Tag}) where Tag = Dual{Tag}.(map(x),-rotation(map.θ)*(map.V+map.ω×(x-map.x₀-map.xₚ)))

# cross product in 2D and rotation matrix in 2D and 3D
import WaterLily: ×
×(a::Number,b::SVector{2,T}) where T = a*SA[-b[2],b[1]]
rotation(θ::T) where T = SA{T}[cos(θ) sin(θ); -sin(θ) cos(θ)]
rotation(θ::SVector{3,T}) where T = SA{T}[cos(θ[1])*cos(θ[2]) cos(θ[1])*sin(θ[2])*sin(θ[3])-sin(θ[1])*cos(θ[3]) cos(θ[1])*sin(θ[2])*cos(θ[3])+sin(θ[1])*sin(θ[3]);
                                          sin(θ[1])*cos(θ[2]) sin(θ[1])*sin(θ[2])*sin(θ[3])+cos(θ[1])*cos(θ[3]) sin(θ[1])*sin(θ[2])*cos(θ[3])-cos(θ[1])*sin(θ[3]);
                                               -sin(θ[2])                         cos(θ[2])*sin(θ[3])                               cos(θ[2])*cos(θ[3])]
using ConstructionBase
setmap(body::AbstractBody; kwargs...) = try
    setproperties(body,map=setproperties(body.map; kwargs...))
catch
    throw(ArgumentError("Cannot update $body with $kwargs"))
end
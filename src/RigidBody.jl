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
    θ_new = sim.body.θ + sim.body.ω*sim.flow.Δt[end]
    sim.body = update!(sim.body; θ=θ_new)
    # remeasure and step
    step!(sim;remeasure=true)
end
```

"""
struct RigidMap{A<:AbstractVector,R} <: Function
    x₀ :: A   # center of translation
    xₚ :: A   # rotation offset
    V  :: A   # linear velocity of the center
    θ  :: R   # rotation (angle in 2D, euler angles in 3D)
    ω  :: R   # angular velocity (scalar in 2D, vector in 3D)
    function RigidMap(x₀::SVector, xₚ::SVector, V::SVector, θ::R, ω::R) where R
        new{typeof(x₀),R}(x₀, xₚ, V, θ, ω)
    end
end
RigidBody(sdf,x₀,θ;xₚ=zero(x₀),V=zero(x₀),ω=zero(θ)) = AutoBody(sdf,RigidMap(x₀,xₚ,V,θ,ω);compose=true)

function (m::RigidMap)(x::SVector,t=0)::SVector
    return rotation(m.θ)*(x-m.x₀-m.xₚ)+m.xₚ
end

# rigid body velocity
velocity(map::RigidMap, x::SVector, t=0) = map.V .+ map.ω×(x - map.x₀ - map.xₚ)

# cross product in 2D and rotation matrix in 2D and 3D
×(a::T,b::SVector{2,T}) where T = a*SA{T}[-b[2],b[1]]
rotation(θ::T) where T = SA{T}[cos(θ) sin(θ); -sin(θ) cos(θ)]
rotation(θ::SVector{3,T}) where T = SA{T}[cos(θ[1])*cos(θ[2]) cos(θ[1])*sin(θ[2])*sin(θ[3])-sin(θ[1])*cos(θ[3]) cos(θ[1])*sin(θ[2])*cos(θ[3])+sin(θ[1])*sin(θ[3]);
                                          sin(θ[1])*cos(θ[2]) sin(θ[1])*sin(θ[2])*sin(θ[3])+cos(θ[1])*cos(θ[3]) sin(θ[1])*sin(θ[2])*cos(θ[3])-cos(θ[1])*sin(θ[3]);
                                               -sin(θ[2])                         cos(θ[2])*sin(θ[3])                               cos(θ[2])*cos(θ[3])]
function update!(body; x₀=body.map.x₀, V=body.map.V,
                       xₚ=body.map.xₚ, θ=body.map.θ, ω=body.map.ω)
    !isa(body.map, RigidMap) && return nothing
    return RigidBody(body.sdf, x₀, θ; xₚ, V, ω)
end
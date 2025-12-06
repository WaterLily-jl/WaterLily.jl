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
struct RigidMap <: Function
    x₀ :: SVector   # center of translation
    xₚ :: SVector   # rotation offset
    V  :: SVector   # linear velocity of the center
    θ               # rotation (angle in 2D, euler angles in 3D)
    ω               # angular velocity (scalar in 2D, vector in 3D)
end

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
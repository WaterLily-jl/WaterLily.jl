# new body type
struct RigidBody{D,T,F<:Function,A<:AbstractVector} <: AbstractBody
    sdf :: F            # signed distance function
    center :: A         # center of linear motion
    velocity :: A       # linear velocity of the center
    pivot    :: A       # offset of the pivot point compared to center
    rot :: Union{T,A}   # rotation (angle in 2D, euler angles in 3D)
    ω :: Union{T,A}     # angular velocity (scalar in 2D, vector in 3D)
    function RigidBody(sdf,center,velocity,pivot,θ,ω)
        T,D = eltype(center),length(center)
        new{D,T,typeof(sdf),typeof(center)}(sdf,center,velocity,pivot,θ,ω)
    end
end
RigidBody(sdf,center,θ;velocity=zero(center),ω=zero(θ),pivot=zero(center)) = RigidBody(sdf,center,velocity,pivot,θ,ω)
function WaterLily.sdf(body::RigidBody{D,T},x,t=0;kwargs...) where {D,T}
    R = rotation(body.rot)
    return body.sdf(R*(x-body.center-body.pivot)+body.pivot,t)
end
rotation(θ::T) where T = SA{T}[cos(θ) sin(θ); -sin(θ) cos(θ)]
rotation(θ::SVector{3,T}) where T = SA{T}[cos(θ[1])*cos(θ[2]) cos(θ[1])*sin(θ[2])*sin(θ[3])-sin(θ[1])*cos(θ[3]) cos(θ[1])*sin(θ[2])*cos(θ[3])+sin(θ[1])*sin(θ[3]);
                                          sin(θ[1])*cos(θ[2]) sin(θ[1])*sin(θ[2])*sin(θ[3])+cos(θ[1])*cos(θ[3]) sin(θ[1])*sin(θ[2])*cos(θ[3])-cos(θ[1])*sin(θ[3]);
                                               -sin(θ[2])                         cos(θ[2])*sin(θ[3])                               cos(θ[2])*cos(θ[3])]
# cross product and new specialized case in 2D
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

# structure to store Simulation state
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
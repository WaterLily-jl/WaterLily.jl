@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])*0.5
@fastmath quick(u,c,d) = median((5c+2d-u)/6,c,median(10c-9u,c,d))
@fastmath vanLeer(u,c,d) = (c≤min(u,d) || c≥max(u,d)) ? c : c+(d-c)*(c-u)/(d-u)
@inline ϕu(a,I,f,u,λ=quick) = @inbounds u>0 ? u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuP(a,Ip,I,f,u,λ=quick) = @inbounds u>0 ? u*λ(f[Ip],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuL(a,I,f,u,λ=quick) = @inbounds u>0 ? u*ϕ(a,I,f) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuR(a,I,f,u,λ=quick) = @inbounds u<0 ? u*ϕ(a,I,f) : u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I])

@fastmath @inline function div(I::CartesianIndex{m},u) where {m}
    init=zero(eltype(u))
    for i in 1:m
     init += @inbounds ∂(i,I,u)
    end
    return init
end
@fastmath @inline function μddn(I::CartesianIndex{np1},μ,f) where np1
    s = zero(eltype(f))
    for j ∈ 1:np1-1
        s+= @inbounds μ[I,j]*(f[I+δ(j,I)]-f[I-δ(j,I)])
    end
    return 0.5s
end
function median(a,b,c)
    if a>b
        b>=c && return b
        a>c && return c
    else
        b<=c && return b
        a<c && return c
    end
    return a
end

function conv_diff!(r,u,Φ;ν=0.1,perdir=())
    r .= 0.
    N,n = size_u(u)
    for i ∈ 1:n, j ∈ 1:n
        # if it is periodic direction
        tagper = (j in perdir)
        # treatment for bottom boundary with BCs
        lowerBoundary!(r,u,Φ,ν,i,j,N,Val{tagper}())
        # inner cells
        @loop (Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u)) - ν*∂(j,CI(I,i),u);
               r[I,i] += Φ[I]) over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        # treatment for upper boundary with BCs
        upperBoundary!(r,u,Φ,ν,i,j,N,Val{tagper}())
    end
end

# Neumann BC Building block
lowerBoundary!(r,u,Φ,ν,i,j,N,::Val{false}) = @loop r[I,i] += ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),u)) - ν*∂(j,CI(I,i),u) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν,i,j,N,::Val{false}) = @loop r[I-δ(j,I),i] += -ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),u)) + ν*∂(j,CI(I,i),u) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowerBoundary!(r,u,Φ,ν,i,j,N,::Val{true}) = @loop (
    Φ[I] = ϕuP(j,CIj(j,CI(I,i),N[j]-2),CI(I,i),u,ϕ(i,CI(I,j),u)) -ν*∂(j,CI(I,i),u); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν,i,j,N,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)

using EllipsisNotation
"""
    accelerate!(r,t,g,U)

Adds a space-time acceleration field g(i,x,t) and velocity field U(i,x,t) such that `rᵢ += gᵢ(i,x,t)+dUᵢ(i,x,t)/dt` at time `t=sum(dt)` to vector field `r`.
"""
accelerate!(_,_,::Nothing,::Union{Nothing,Tuple}) = nothing
accelerate!(r,t,f) = for i ∈ 1:last(size(r))
    @loop r[I,i] += f(i,loc(i,I,eltype(r)),t) over I ∈ CartesianIndices(Base.front(size(r)))
end
accelerate!(r,t,g::Function,::Union{Nothing,Tuple}) = accelerate!(r,t,g)
accelerate!(r,t,::Nothing,U::Function) = accelerate!(r,t,(i,x,t)->ForwardDiff.derivative(τ->U(i,x,τ),t))
accelerate!(r,t,g::Function,U::Function) = accelerate!(r,t,(i,x,t)->g(i,x,t)+ForwardDiff.derivative(τ->U(i,x,τ),t))
"""
    Flow{D::Int, T::Float, Sf<:AbstractArray{T,D}, Vf<:AbstractArray{T,D+1}, Tf<:AbstractArray{T,D+2}}

Composite type for a multidimensional immersed boundary flow simulation.

Flow solves the unsteady incompressible [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) on a Cartesian grid.
Solid boundaries are modelled using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/).
The primary variables are the scalar pressure `p` (an array of dimension `D`)
and the velocity vector field `u` (an array of dimension `D+1`).
"""
struct Flow{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Tf<:AbstractArray{T}}
    # Fluid fields
    u :: Vf # velocity vector field
    u⁰:: Vf # previous velocity
    f :: Vf # force vector
    p :: Sf # pressure scalar field
    σ :: Sf # divergence scalar
    # BDIM fields
    V :: Vf # body velocity vector
    μ₀:: Vf # zeroth-moment vector
    μ₁:: Tf # first-moment tensor field
    # Non-fields
    U :: Union{NTuple{D,Number},Function} # domain boundary values
    Δt:: Vector{T} # time step (stored in CPU memory)
    ν :: T # kinematic viscosity
    g :: Union{Function,Nothing} # (possibly time-varying) uniform acceleration field
    exitBC :: Bool # Convection exit
    perdir :: NTuple # tuple of periodic direction
    function Flow(N::NTuple{D}, U; f=Array, Δt=0.25, ν=0., g=nothing,
                  uλ::Function=(i, x) -> 0., perdir=(), exitBC=false, T=Float32) where D
        Ng = N .+ 2
        Nd = (Ng..., D)
        u = Array{T}(undef, Nd...) |> f; apply!(uλ, u);
        BC!(u,U,exitBC,perdir); exitBC!(u,u,0.)
        u⁰ = copy(u);
        fv, p, σ = zeros(T, Nd) |> f, zeros(T, Ng) |> f, zeros(T, Ng) |> f
        V, μ₀, μ₁ = zeros(T, Nd) |> f, ones(T, Nd) |> f, zeros(T, Ng..., D, D) |> f
        BC!(μ₀,ntuple(zero, D),false,perdir)
        new{D,T,typeof(p),typeof(u),typeof(μ₁)}(u,u⁰,fv,p,σ,V,μ₀,μ₁,U,T[Δt],ν,g,exitBC,perdir)
    end
end

"""
    time(a::Flow)

Current flow time.
"""
time(a::Flow) = sum(@view(a.Δt[1:end-1]))

function BDIM!(a::Flow)
    dt = a.Δt[end]
    @loop a.f[Ii] = a.u⁰[Ii]+dt*a.f[Ii]-a.V[Ii] over Ii in CartesianIndices(a.f)
    @loop a.u[Ii] += μddn(Ii,a.μ₁,a.f)+a.V[Ii]+a.μ₀[Ii]*a.f[Ii] over Ii ∈ inside_u(size(a.p))
end

function project!(a::Flow{n},b::AbstractPoisson,w=1) where n
    dt = w*a.Δt[end]
    @inside b.z[I] = div(I,a.u); b.x .*= dt # set source term & solution IC
    solver!(b)
    for i ∈ 1:n  # apply solution and unscale to recover pressure
        @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)
    end
    b.x ./= dt
end

"""
    mom_step!(a::Flow,b::AbstractPoisson)

Integrate the `Flow` one time step using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/)
and the `AbstractPoisson` pressure solver to project the velocity onto an incompressible flow.
"""
@fastmath function mom_step!(a::Flow{N},b::AbstractPoisson;udf=nothing,kwargs...) where N
    a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    @log "p"
    conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν,perdir=a.perdir)
    udf!(a,udf,t₀; kwargs...)
    accelerate!(a.f,t₀,a.g,a.U)
    BDIM!(a); BC!(a.u,a.U,a.exitBC,a.perdir,t₁) # BC MUST be at t₁
    a.exitBC && exitBC!(a.u,a.u⁰,a.Δt[end]) # convective exit
    project!(a,b); BC!(a.u,a.U,a.exitBC,a.perdir,t₁)
    # corrector u → u¹
    @log "c"
    conv_diff!(a.f,a.u,a.σ,ν=a.ν,perdir=a.perdir)
    udf!(a,udf,t₁; kwargs...)
    accelerate!(a.f,t₁,a.g,a.U)
    BDIM!(a); scale_u!(a,0.5); BC!(a.u,a.U,a.exitBC,a.perdir,t₁)
    project!(a,b,0.5); BC!(a.u,a.U,a.exitBC,a.perdir,t₁)
    push!(a.Δt,CFL(a))
end
scale_u!(a,scale) = @loop a.u[Ii] *= scale over Ii ∈ inside_u(size(a.p))

function CFL(a::Flow;Δt_max=10)
    @inside a.σ[I] = flux_out(I,a.u)
    min(Δt_max,inv(maximum(a.σ)+5a.ν))
end
@fastmath @inline function flux_out(I::CartesianIndex{d},u) where {d}
    s = zero(eltype(u))
    for i in 1:d
        s += @inbounds(max(0.,u[I+δ(i,I),i])+max(0.,-u[I,i]))
    end
    return s
end

"""
    udf!(flow::Flow,udf::Function,t)

User defined function using `udf::Function` to operate on `flow::Flow` during the predictor and corrector step, in sync with time `t`.
Keyword arguments must be passed to `sim_step!` for them to be carried over the actual function call.
"""
udf!(_,::Nothing,_) = nothing
udf!(flow::Flow,force!::Function,t; kwargs...) = force!(flow,t; kwargs...)

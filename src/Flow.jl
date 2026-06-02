@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])/2
@fastmath quick(u,c,d) = median((5c+2d-u)/6,c,median(10c-9u,c,d))
@fastmath vanLeer(u,c,d) = (c≤min(u,d) || c≥max(u,d)) ? c : c+(d-c)*(c-u)/(d-u)
@fastmath cds(u,c,d) = (c+d)/2

@inline ϕu(a,I,f,u,λ) = @inbounds u>0 ? u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuP(a,Ip,I,f,u,λ) = @inbounds u>0 ? u*λ(f[Ip],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuL(a,I,f,u,λ) = @inbounds u>0 ? u*ϕ(a,I,f) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@inline ϕuR(a,I,f,u,λ) = @inbounds u<0 ? u*ϕ(a,I,f) : u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I])

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

# Cell-centred and face-averaged effective-viscosity lookups. A scalar
# ν dispatches to the original constant-ν kernel (zero overhead). A
# callable `ν(I)` returns the cell-centred effective viscosity, computed
# on the fly in the flux — no stored νₑ array. `_νf` linearly
# interpolates the closure to the cell face so the diffusive flux stays
# consistent for inhomogeneous ν. The closure may itself wrap a
# downstream array (e.g. a VoF volume-fraction field) if a non-local
# model ever needs one.
@inline _ν(ν::Number,I) = ν
@inline _ν(ν,I) = ν(I)
@inline _νf(ν::Number,j,I) = ν
@inline _νf(ν,j,I) = @inbounds (ν(I) + ν(I-δ(j,I))) / 2

"""
    conv_diff!(r, u, Φ, λ; ν=0.1, perdir=())

Compute the convective + diffusive momentum residual

```
r[I, i] = -∂_j ( u_j u_i - ν_face · ∂_j u_i )
```

for the face-staggered velocity `u`. The high-order convective flux is
limited by `λ` (e.g. `quick`, `vanLeer`); the diffusive flux uses the
face-averaged effective viscosity (see `_νf`). `Φ` is a per-cell
workspace array. `ν` may be a scalar (uniform Re) or a callable `ν(I)`
returning the cell-centred effective viscosity (variable Re for VoF /
LES), evaluated on the fly with no stored array. Directions listed in
`perdir` are treated as periodic at both boundaries.
"""
function conv_diff!(r,u,Φ,λ::F;ν=0.1,perdir=()) where {F}
    r .= zero(eltype(r))
    N,n = size_u(u)
    for i ∈ 1:n, j ∈ 1:n
        # if it is periodic direction
        tagper = (j in perdir)
        # treatment for bottom boundary with BCs
        lowerBoundary!(r,u,Φ,ν,i,j,N,λ,Val{tagper}())
        # inner cells
        @loop (Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) - _νf(ν,j,I)*∂(j,CI(I,i),u);
               r[I,i] += Φ[I]) over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        # treatment for upper boundary with BCs
        upperBoundary!(r,u,Φ,ν,i,j,N,λ,Val{tagper}())
    end
end

# Neumann BC Building block
lowerBoundary!(r,u,Φ,ν,i,j,N,λ,::Val{false}) = @loop r[I,i] += ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) - _νf(ν,j,I)*∂(j,CI(I,i),u) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν,i,j,N,λ,::Val{false}) = @loop r[I-δ(j,I),i] += -ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) + _νf(ν,j,I)*∂(j,CI(I,i),u) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowerBoundary!(r,u,Φ,ν,i,j,N,λ,::Val{true}) = @loop (
    Φ[I] = ϕuP(j,CIj(j,CI(I,i),N[j]-2),CI(I,i),u,ϕ(i,CI(I,j),u),λ) -_νf(ν,j,I)*∂(j,CI(I,i),u); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν,i,j,N,λ,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)

"""
    accelerate!(r,t,g,U)

Accounts for applied and reference-frame acceleration using `rᵢ += g(i,x,t)+dU(i,x,t)/dt`
"""
accelerate!(r,t,::Nothing,::Union{Nothing,Tuple}) = nothing
accelerate!(r,t,f::Function) = @loop r[Ii] += f(last(Ii),loc(Ii,eltype(r)),t) over Ii ∈ CartesianIndices(r)
accelerate!(r,t,g::Function,::Union{Nothing,Tuple}) = accelerate!(r,t,g)
accelerate!(r,t,::Nothing,U::Function) = accelerate!(r,t,(i,x,t)->derivative(τ->U(i,x,τ),t))
accelerate!(r,t,g::Function,U::Function) = accelerate!(r,t,(i,x,t)->g(i,x,t)+derivative(τ->U(i,x,τ),t))

abstract type AbstractFlow{D,T} end
"""
    Flow{D::Int, T::Float, Sf<:AbstractArray{T,D}, Vf<:AbstractArray{T,D+1}, Tf<:AbstractArray{T,D+2}}

Composite type for a multidimensional immersed boundary flow simulation.

Flow solves the unsteady incompressible [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) on a Cartesian grid.
Solid boundaries are modelled using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/).
The primary variables are the scalar pressure `p` (an array of dimension `D`)
and the velocity vector field `u` (an array of dimension `D+1`).
"""
struct Flow{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Tf<:AbstractArray{T}, Nf} <: AbstractFlow{D,T}
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
    uBC :: Union{NTuple{D,Number},Function} # domain boundary values/function
    Δt:: Vector{T} # time step (stored in CPU memory)
    ν :: Nf # kinematic viscosity: scalar `T` (default) or a cell-centred closure `ν(I)`
    g :: Union{Function,Nothing} # acceleration field funciton
    exitBC :: Bool # Convection exit
    perdir :: NTuple # tuple of periodic direction
    function Flow(N::NTuple{D}, uBC; mem=Array, Δt=0.25, ν=0., g=nothing,
            uλ=nothing, perdir=(), exitBC=false, T=Float32) where D
        Ng = N .+ 2
        Nd = (Ng..., D)
        isnothing(uλ) && (uλ = ic_function(uBC))
        u = Array{T}(undef, Nd...) |> mem
        isa(uλ, Function) ? apply!(uλ, u) : apply!((i,x)->uλ[i], u)
        BC!(u,uBC,exitBC,perdir); exitBC!(u,u,0.)
        u⁰ = copy(u)
        fv, p, σ = zeros(T, Nd) |> mem, zeros(T, Ng) |> mem, zeros(T, Ng) |> mem
        V, μ₀, μ₁ = zeros(T, Nd) |> mem, ones(T, Nd) |> mem, zeros(T, Ng..., D, D) |> mem
        BC!(μ₀,ntuple(zero, D),false,perdir)
        # A scalar ν is stored as `T`; a callable closure is stored by
        # reference, so downstream code that mutates the closure's own
        # state (e.g. a VoF field) stays in sync across `mom_step!`.
        ν_store = isa(ν, Function) ? ν : T(ν)
        new{D,T,typeof(p),typeof(u),typeof(μ₁),typeof(ν_store)}(u,u⁰,fv,p,σ,V,μ₀,μ₁,uBC,T[Δt],ν_store,g,exitBC,perdir)
    end
end

"""
    mom_step!(a::AbstractFlow,b::AbstractPoisson;λ=quick,udf=nothing,kwargs...)

Integrate the `Flow` one time step using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/)
and the `AbstractPoisson` pressure solver to project the velocity onto an incompressible flow.
"""
@fastmath function mom_step!(a::AbstractFlow,b::AbstractPoisson;λ=quick,udf=nothing,kwargs...)
    a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    @log "p"
    mom_predict!(a,t₀,t₁;λ,udf,kwargs...)
    mom_project!(a,b,1,t₁)
    # corrector u → u¹
    @log "c"
    mom_correct!(a,t₁;λ,udf,kwargs...)
    mom_project!(a,b,0.5,t₁)
    push!(a.Δt,CFL(a))
end

"""
    time(a::AbstractFlow)

Current flow time.
"""
time(a::AbstractFlow) = sum(@view(a.Δt[1:end-1]))

function BDIM!(a::AbstractFlow)
    dt = a.Δt[end]
    @loop a.f[Ii] = a.u⁰[Ii]+dt*a.f[Ii]-a.V[Ii] over Ii in CartesianIndices(a.f)
    @loop a.u[Ii] += μddn(Ii,a.μ₁,a.f)+a.V[Ii]+a.μ₀[Ii]*a.f[Ii] over Ii ∈ inside_u(size(a.p))
end

"""
    mom_predict!(a::AbstractFlow, t₀, t₁; λ=quick, udf=nothing, kwargs...)

Predictor phase of `mom_step!`: advect under `u⁰`, apply BDIM, enforce BCs.
On return `a.u` is BC-consistent and ready for pressure projection.
`t₀` and `t₁` are the start and end times of the step; BCs are enforced at the
end-of-step time `sum(a.Δt)`.
"""
function mom_predict!(a::AbstractFlow, t₀, t₁; λ=quick, udf=nothing, kwargs...)
    conv_diff!(a.f,a.u⁰,a.σ,λ;ν=a.ν,perdir=a.perdir)
    udf!(a,udf,t₀; kwargs...)
    accelerate!(a.f,t₀,a.g,a.uBC)
    BDIM!(a); BC!(a.u,a.uBC,a.exitBC,a.perdir,t₁) # BC MUST be at t₁
    a.exitBC && exitBC!(a.u,a.u⁰,a.Δt[end]) # convective exit
end

"""
    mom_correct!(a::AbstractFlow, t; λ=quick, udf=nothing, kwargs...)

Corrector phase of `mom_step!`: advect under the projected `u`, apply BDIM,
blend with the trapezoidal weight, enforce BCs at time-step end-time `t`.
On return `a.u` is BC-consistent and ready for pressure projection.
"""
function mom_correct!(a::AbstractFlow, t; λ=quick, udf=nothing, kwargs...)
    conv_diff!(a.f,a.u,a.σ,λ;ν=a.ν,perdir=a.perdir)
    udf!(a,udf,t; kwargs...)
    accelerate!(a.f,t,a.g,a.uBC)
    BDIM!(a); scale_u!(a,0.5); BC!(a.u,a.uBC,a.exitBC,a.perdir,t)
end
scale_u!(a,scale) = @loop a.u[Ii] *= scale over Ii ∈ inside_u(size(a.p))

"""
    mom_project!(a::AbstractFlow, b::AbstractPoisson, w, t)

Projection phase of `mom_step!`: solve the pressure Poisson equation, correct
the velocity by `w·Δt·∇p`, and re-enforce BCs.
On return `a.u` is divergence-free and BC-consistent.
"""
function mom_project!(a::AbstractFlow, b::AbstractPoisson, w, t)
    dt = w*a.Δt[end]
    @inside b.z[I] = div(I,a.u); b.x .*= dt # set source term & solution IC
    solver!(b)
    for i ∈ 1:ndims(a.p)  # apply solution and unscale to recover pressure
        @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)
    end
    b.x ./= dt
    BC!(a.u,a.uBC,a.exitBC,a.perdir,t)
end

# Maximum effective viscosity for the CFL viscous limit. A scalar
# returns itself (zero overhead). A closure is evaluated into the
# scratch buffer `σ` and reduced over the interior — called only after
# the flux-out maximum has already been read out of `σ`.
@inline _ν_max(ν::Number, σ) = ν
function _ν_max(ν, σ)
    @inside σ[I] = ν(I)
    maximum(@view σ[inside(σ)])
end

function CFL(a::AbstractFlow;Δt_max=10)
    @inside a.σ[I] = flux_out(I,a.u)
    fluxmax = maximum(a.σ)
    min(Δt_max,inv(fluxmax+5*_ν_max(a.ν,a.σ)))
end
@fastmath @inline function flux_out(I::CartesianIndex{d},u) where {d}
    s = zero(eltype(u))
    for i in 1:d
        s += @inbounds(max(0,u[I+δ(i,I),i])+max(0,-u[I,i]))
    end
    return s
end

"""
    udf!(flow::AbstractFlow,udf::Function,t)

User defined function using `udf::Function` to operate on `flow::AbstractFlow` during the predictor and corrector step, in sync with time `t`.
Keyword arguments must be passed to `sim_step!` for them to be carried over the actual function call.
"""
udf!(flow,::Nothing,t; kwargs...) = nothing
udf!(flow,force!::Function,t; kwargs...) = force!(flow,t; kwargs...)

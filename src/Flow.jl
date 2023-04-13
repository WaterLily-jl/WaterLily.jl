@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])*0.5
@fastmath quick(u,c,d) = median((5c+2d-u)/6,c,median(10c-9u,c,d))
@fastmath vanLeer(u,c,d) = (c≤min(u,d) || c≥max(u,d)) ? c : c+(d-c)*(c-u)/(d-u)
@inline ϕu(a,I,f,u,λ=quick) = @inbounds u>0 ? u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@fastmath @inline div(I::CartesianIndex{m},u) where {m} = sum(@inbounds ∂(i,I,u) for i ∈ 1:m)
@fastmath @inline function div_operator(I::CartesianIndex{m},u) where {m}
    init=zero(eltype(u))
    for i in 1:m
     init += @inbounds ∂(i,I,u)
    end
    return init
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

@fastmath function tracer_transport!(r,f,u,Φ;Pe=0.1)
    N,n = size_u(u)
    for j ∈ 1:n
        @loop r[I] += ϕ(j,I,f)*u[I,j]-Pe*∂(j,I,f) over I ∈ slice(N,2,j,2)
        @loop Φ[I] = ϕu(j,I,f,u[I,j])-Pe*∂(j,I,f) over I ∈ inside_u(N,j)
        @loop r[I] += Φ[I] over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I)] -= Φ[I] over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I)] -= ϕ(j,I,f)*u[I,j]-Pe*∂(j,I,f) over I ∈ slice(N,N[j],j,2)
    end
end

@fastmath function conv_diff!(r,u,Φ;ν=0.1)
    r .= 0.
    N,n = size_u(u)
    for i ∈ 1:n, j ∈ 1:n
        @loop r[I,i] = r[I,i] + ϕ(j,CI(I,i),u)*ϕ(i,CI(I,j),u)-ν*∂(j,CI(I,i),u) over I ∈ slice(N,2,j,2)
        @loop Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u))-ν*∂(j,CI(I,i),u) over I ∈ inside_u(N,j)
        @loop r[I,i] = r[I,i] + Φ[I] over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] = r[I-δ(j,I),i] - Φ[I] over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] = r[I-δ(j,I),i] - ϕ(j,CI(I,i),u)*ϕ(i,CI(I,j),u) + ν*∂(j,CI(I,i),u) over I ∈ slice(N,N[j],j,2)
    end
end

"""
    Flow{D, V, S, F, B, T}

Composite type for a multidimensional immersed boundary flow simulation.

Flow solves the unsteady incompressible [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) on a Cartesian grid.
Solid boundaries are modelled using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/).
The primary variables are the scalar pressure `p` (an array of dimension `N`)
and the velocity vector field `u` (an array of dimension `M=N+1`).
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
    σᵥ:: Sf # body velocity divergence
    μ₀:: Vf # zeroth-moment vector
    μ₁:: Tf # first-moment tensor field
    # Non-fields
    U :: NTuple{D, T} # domain boundary values
    Δt:: Vector{T} # time step (stored in CPU memory)
    ν :: T # kinematic viscosity
    function Flow(N::NTuple{D}, U::NTuple{D}; f=identity, Δt=0.25, ν=0., uλ::Function=(i, x) -> 0., T=Float64) where D
        Ng = N .+ 2
        Nd = (Ng..., D)
        @assert length(U) == D
        u = Array{T}(undef, Nd...) |> f
        apply!(uλ, u)
        BC!(u, U)
        u⁰ = copy(u)
        fv, p, σ = zeros(T, Nd) |> f, zeros(T, Ng) |> f, zeros(T, Ng) |> f
        V, σᵥ = zeros(T, Nd) |> f, zeros(T, Ng) |> f
        μ₀ = ones(T, Nd) |> f
        BC!(μ₀, tuple(zeros(T, D)...))
        μ₁ = zeros(T, Ng..., D, D) |> f
        new{D,T,typeof(p),typeof(u),typeof(μ₁)}(u,u⁰,fv,p,σ,V,σᵥ,μ₀,μ₁,U,T[Δt],ν)
    end
end

@fastmath function BDIM!(a::Flow{n}) where n
    @. a.f = a.u⁰+a.Δt[end]*a.f-a.V
    for j ∈ 1:n, i ∈ 1:n
        @loop a.u[I,i] = a.u[I,i] + μddn(j,CI(I,i),a.μ₁,a.f) over I ∈ inside(a.p)
    end
    @. a.u += a.V+a.μ₀*a.f
end
@inline μddn(j,I::CartesianIndex,μ,f) = @inbounds 0.5μ[I,j]*(f[I+δ(j,I)]-f[I-δ(j,I)])

@fastmath function project!(a::Flow{n},b::AbstractPoisson,w=1) where n
    dt = a.Δt[end]
    @inside a.σ[I] = (div_operator(I,a.u)+w*a.σᵥ[I])/dt
    solver!(b,a.σ)
    for i ∈ 1:n
        @loop a.u[I,i] = a.u[I,i] - dt*a.μ₀[I,i]*∂(i,I,a.p) over I ∈ inside(a.σ)
    end
end

"""
    mom_step!(a::Flow,b::AbstractPoisson)

Integrate the `Flow` one time step using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/)
and the `AbstractPoisson` pressure solver to project the velocity onto an incompressible flow.
"""
@fastmath function mom_step!(a::Flow,b::AbstractPoisson)
    a.u⁰ .= a.u; a.u .= 0
    # predictor u → u'
    conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν)
    BDIM!(a); BC!(a.u,a.U)
    project!(a,b); BC!(a.u,a.U)
    # corrector u → u¹
    conv_diff!(a.f,a.u,a.σ,ν=a.ν)
    BDIM!(a); BC!(a.u,a.U,2)
    project!(a,b,2); a.u ./= 2; BC!(a.u,a.U)
    push!(a.Δt,CFL(a))
end

function CFL(a::Flow)
    @inside a.σ[I] = flux_out(I,a.u)
    min(10.,inv(maximum(a.σ)+5a.ν))
end
@fastmath @inline function flux_out(I::CartesianIndex{d},u) where {d}
    s = zero(eltype(u))
    for i in 1:d
        s += @inbounds(max(0.,u[I+δ(i,I),i])+max(0.,-u[I,i]))
    end
    return s
end
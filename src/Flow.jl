@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])*0.5
@fastmath quick(u,c,d) = median((5c+2d-u)/6,c,median(10c-9u,c,d))
@fastmath vanLeer(u,c,d) = (c≤min(u,d) || c≥max(u,d)) ? c : c+(d-c)*(c-u)/(d-u)
@inline ϕu(a,I,f,u,λ=quick) = @inbounds u>0 ? u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@fastmath @inline div(I::CartesianIndex{m},u) where {m} = sum(@inbounds ∂(i,I,u) for i ∈ 1:m)
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
        @loop r[I,i] += ϕ(j,CI(I,i),u)*ϕ(i,CI(I,j),u)-ν*∂(j,CI(I,i),u) over I ∈ slice(N,2,j,2)
        @loop Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u))-ν*∂(j,CI(I,i),u) over I ∈ inside_u(N,j)
        @loop r[I,i] += Φ[I] over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= ϕ(j,CI(I,i),u)*ϕ(i,CI(I,j),u)-ν*∂(j,CI(I,i),u) over I ∈ slice(N,N[j],j,2)
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
struct Flow{D, V, S, F, B, T}
    # Fluid fields
    u :: V # velocity vector
    u⁰:: V # previous velocity
    f :: V # force vector
    p :: S # pressure scalar
    σ :: S # divergence scalar
    # BDIM fields
    V :: V # body velocity vector
    σᵥ:: S # body velocity divergence
    μ₀:: V # zeroth-moment on faces
    μ₁:: F # first-moment vector on faces
    # Non-fields
    bc:: B # ghost-donor boundary cell indices
    U :: NTuple{D, T} # domain boundary values
    Δt:: Vector{T} # time step (stored in CPU memory)
    ν :: T # kinematic viscosity
    function Flow(N::NTuple{D}, U::NTuple{D}; Δt=0.25, ν=0., uλ::Function=(i, x) -> 0., f=identity, T=Float64) where D
        Ng = N .+ 2
        Nd = (Ng..., D)
        @assert length(U) == D
        u = Array{T}(undef, Nd...) |> OA(D) |> f
        apply!(uλ, u)

        bc = WaterLily.bc_indices(Ng) |> f
        BC!(u, U, bc)
        u⁰ = copy(u)
        fv, p, σ = zeros(T, Nd) |> OA(D) |> f, zeros(T, Ng) |> OA() |> f, zeros(T, Ng) |> OA() |> f
        V, σᵥ = zeros(T, Nd) |> OA(D) |> f, zeros(T, Ng) |> OA() |> f

        μ₀ = ones(T, Nd) |> OA(D) |> f
        BC!(μ₀, tuple(zeros(T, D)...), bc)
        μ₁ = zeros(T, Ng..., D, D) |> OA(D, 2) |> f

        new{D,typeof(u),typeof(p),typeof(μ₁),typeof(bc),T}(u,u⁰,fv,p,σ,V,σᵥ,μ₀,μ₁,bc,U,T[Δt],ν)
    end
end

@fastmath function BDIM!(a::Flow{n}) where n
    @. a.f = a.u⁰+a.Δt[end]*a.f-a.V
    for j ∈ 1:n, i ∈ 1:n
        @loop a.u[I,i] += μddn(j,CI(I,i),a.μ₁,a.f) over I ∈ inside(a.p)
    end
    @. a.u += a.V+a.μ₀*a.f
end
@inline μddn(j,I::CartesianIndex,μ,f) = @inbounds 0.5μ[I,j]*(f[I+δ(j,I)]-f[I-δ(j,I)])

@fastmath function project!(a::Flow{n},b::AbstractPoisson{n},w=1) where n
    @inside a.σ[I] = (div(I,a.u)+w*a.σᵥ[I])/a.Δt[end]
    solver!(a.p,b,a.σ)
    for i ∈ 1:n
        @loop a.u[I,i] -= a.Δt[end]*a.μ₀[I,i]*∂(i,I,a.p) over I ∈ inside(a.σ)
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
    _ENABLE_PUSH && push!(a.Δt,CFL(a))
end

function CFL(a::Flow{n}) where n
    mx = maximum(fout(I,a.u) for I ∈ inside(a.p))
    min(10.,inv(mx+5a.ν))
end
@fastmath @inline fout(I::CartesianIndex{d},u) where {d} =
    sum(@inbounds(max(0.,u[I+δ(a,I),a])+max(0.,-u[I,a])) for a ∈ 1:d)

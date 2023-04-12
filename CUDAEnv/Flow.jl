using WaterLily
using WaterLily: δ, slice, apply!, BC!
using CUDA: cu

@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]

struct Flow{D, V, S, F, T}
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
    U :: NTuple{D, T} # domain boundary values
    Δt:: Vector{T} # time step (stored in CPU memory)
    ν :: T # kinematic viscosity
    function Flow(N::NTuple{D}, U::NTuple{D}; Δt=0.25, ν=0., uλ::Function=(i, x) -> 0., f=identity, T=Float64) where D
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

        new{D,typeof(u),typeof(p),typeof(μ₁),T}(u,u⁰,fv,p,σ,V,σᵥ,μ₀,μ₁,U,T[Δt],ν)
    end
end

# main
N = (3, 4)
flow = Flow(N, (1.0, 1.0); f = cu, T = Float64);
using WaterLily
using WaterLily: OA, δ, slice, apply!, BC!
using CUDA

@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]

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
    bc:: B
    U :: NTuple{D, T}  # domain boundary values
    Δt:: Vector{T}  # time step
    ν :: T                  # kinematic viscosity
    function Flow(N::NTuple{D}, U; Δt=0.25, ν=0., uλ::Function=(i, x) -> 0., f=identity, T=Float64) where D
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

# main
N = (3, 4)
flow = Flow(N, (1.0, 0.0); f = identity, T = Float64);
return nothing
using KernelAbstractions, Adapt, OffsetArrays, BenchmarkTools, OutMacro

if Base.find_package("CUDA") !== nothing
    using CUDA
    using CUDA.CUDAKernels
    const backend = CUDABackend()
    CUDA.allowscalar(false)
else
    const backend = CPU()
end

@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]

# Utils (to be moved to utils.jl)
O(D=0, d=1) = OffsetArrays.Origin(D > 0 ? (zeros(Int, D)..., ones(Int, d)...) : 0)
function slice(dims::NTuple{N}, i, j, low = 1, trim = 0) where N
    CartesianIndices(ntuple(k-> k == j ? (i:i) : (low:dims[k] - trim), N))
end
@inline CI(a...) = CartesianIndex(a...)
@inline δ(i,N::Int) = CI(ntuple(j -> j==i ? 1 : 0, N))
@inline δ(i,I::CartesianIndex{N}) where {N} = δ(i,N)
adapt!(u) = backend == CPU() ? u : adapt(CuArray, u)

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
    function Flow(N::NTuple{D}, U; Δt=0.25, ν=0., uλ::Function=(i, x) -> 0., T=Float64) where D
        Ng = N .+ 2
        Nd = (Ng..., D)
        @assert length(U) == D
        u = Array{T}(undef, Nd...) |> O(D) |> adapt!
        # apply!(uλ, u) # not working yet, TODO

        bc_list = Tuple{CartesianIndex, CartesianIndex, Int}[]
        for d ∈ 1:D
            slice_ghost_start = slice(Ng, 0, d, 1, 2)
            slice_donor_start = slice_ghost_start .+ δ(d, D)
            slice_ghost_end = slice(Ng, Ng[d] - 1, d, 1, 2)
            slice_donor_end = slice_ghost_end .- δ(d, D)
            push!(bc_list, zip(slice_ghost_start, slice_donor_start, ntuple(x -> d, length(slice_ghost_start)))...,
                zip(slice_ghost_end, slice_donor_end, ntuple(x -> d, length(slice_ghost_end)))...)
        end
        bc = Tuple.(bc_list) |> adapt!

        BC!(u, U, bc)
        u⁰ = copy(u)
        f, p, σ = zeros(T, Nd) |> O(D) |> adapt!, zeros(T, Ng) |> O() |> adapt!, zeros(T, Ng) |> O() |> adapt!
        V, σᵥ = zeros(T, Nd) |> O(D) |> adapt!, zeros(T, Ng) |> O() |> adapt!

        μ₀ = ones(T, Nd) |> O(D) |> adapt!
        BC!(μ₀, tuple(zeros(T, D)...), bc)
        μ₁ = zeros(T, Ng..., D, D) |> O(D, 2) |> adapt!

        new{D,typeof(u),typeof(p),typeof(μ₁),typeof(bc),T}(u,u⁰,f,p,σ,V,σᵥ,μ₀,μ₁,bc,U,T[Δt],ν)
    end
end


# Apply boundary conditions to the ghost cells (bc) of a _vector_ field
function BC!(u, U, bc, f = 1.0)
    _BC!(backend, 64)(u, U, bc, f, ndrange=size(bc))
end
@kernel function _BC!(u, @Const(U), @Const(bc), @Const(f))
    i = @index(Global, Linear)
    ghostI, donorI, di = bc[i][1], bc[i][2], bc[i][3]
    # _, D = size_u(u)
    for d ∈ 1:2
        if d == di
            u[ghostI, d] = f * U[d]
        else
            u[ghostI, d] = u[donorI, d]
        end
    end
end
# Apply boundary conditions to the ghost cells (bc) of a _scalar_ field
function BC!(u, bc)
    _BC!(backend, 64)(u, bc, ndrange=size(bc))
end
@kernel function _BC!(u, @Const(bc))
    i = @index(Global, Linear)
    ghostI, donorI = bc[i][1], bc[i][2]
    u[ghostI] = u[donorI]
end

# main
N = (3, 4)
flow = Flow(N, (1.0, 0.0); T = Float64);
return nothing
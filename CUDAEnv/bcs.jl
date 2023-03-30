using KernelAbstractions, Adapt, OffsetArrays, BenchmarkTools

if Base.find_package("CUDA") !== nothing
    using CUDA
    using CUDA.CUDAKernels
    const backend = CUDABackend()
    CUDA.allowscalar(false)
else
    const backend = CPU()
end

# helper functions
ArrayT = (backend == CPU()) ? Array : CuArray
splitn(n) = Base.front(n), n[end]
size_u(u) = splitn(size(u))
@inline δ(i, N::Int) = CartesianIndex(ntuple(j -> j == i ? 1 : 0, N))
@inline δ(i, I::CartesianIndex{N}) where {N} = δ(i, N)
@inline ∂(a, I::CartesianIndex{d}, f::AbstractArray{T,d}) where {T, d} = @inbounds f[I] - f[I - δ(a, I)]
@inline ∂(a, I::CartesianIndex{m}, u::AbstractArray{T,n}) where {T, n, m} = @inbounds u[I + δ(a, I), a] - u[I, a]
origin0(D=0) = OffsetArrays.Origin(D > 0 ? (zeros(Int, D)..., 1) : 0)
function slice(dims::NTuple{N}, i, j, low = 1, trim = 0) where N
    CartesianIndices(ntuple(k-> k == j ? (i:i) : (low:dims[k] - trim), N))
end

# flow and kernels
struct Flow{B <: Backend, V, P, UBC, BC}
    backend :: B
    u :: V
    σ :: P
    U :: UBC
    bcs :: BC
end

function Flow(N::NTuple{D}, U; backend = CPU(), ftype = Float32) where D
    u = rand(ftype, ((N .+ 2)..., D)) |> origin0(D)
    σ = Array{ftype}(undef, N .+ 2) |> origin0()
    bcs, Ng = [], size(σ)
    for d ∈ 1:D
        slice_ghost_start = slice(Ng, 0, d, 1, 2)
        slice_donor_start = slice_ghost_start .+ δ(d, D)
        slice_ghost_end = slice(Ng, Ng[d] - 1, d, 1, 2)
        slice_donor_end = slice_ghost_end .- δ(d, D)
        push!(bcs, zip(slice_ghost_start, slice_donor_start, ntuple(x -> d, length(slice_ghost_start)))...,
            zip(slice_ghost_end, slice_donor_end, ntuple(x -> d, length(slice_ghost_end)))...)
    end
    return Flow(backend, adapt(ArrayT, u), adapt(ArrayT, σ), U, adapt(ArrayT, Tuple.(bcs)))
end

# wrapper for boundary conditions kernel
function bcs!(flow, f = 1.0)
    _bcs!(backend, 64)(flow.u, flow.U, f, flow.bcs, ndrange=size(flow.bcs))
end

# bounary conditions kernel
@kernel function _bcs!(u, @Const(U), @Const(f), @Const(bcs))
    i = @index(Global, Linear)
    ghostI, donorI, di = bcs[i][1], bcs[i][2], bcs[i][3]
    _, D = size_u(u)
    for d ∈ 1:D
        if d == di
            u[ghostI, d] = f * U[d]
        else
            u[ghostI, d] = u[donorI, d]
        end
    end
end

# main
const FT = Float32
N = (3, 4)

flow = Flow(N, (1.0, 0.0, 0.0); backend = backend, ftype = FT)
@btime bcs!($flow)
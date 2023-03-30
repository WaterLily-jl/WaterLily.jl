struct ClampedView{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent :: A
    @inline ClampedView(parent::AbstractArray{T,N}) where {T,N} = new{T,N,typeof(parent)}(parent)
end
Base.parent(A::ClampedView) = A.parent
Base.size(A::ClampedView) = size(parent(A))
@inline Base.getindex(A::ClampedView{<:Any,N}, I::Vararg{Int,N}) where N = @inbounds parent(A)[clamp.(I,axes(parent(A)))...]

@inline δ(i) = CartesianIndex(ntuple(j -> j==i ? 1 : 0, 2))
laplace!(b,a) = @inbounds for I ∈ CartesianIndices(b)
    b[I] = a[I-δ(1)]+a[I+δ(1)]+a[I-δ(2)]+a[I+δ(2)]-4a[I]
end

N = (2^10,2^10)
a = ClampedView(rand(Float32,N));
a[CartesianIndex(-1,1)] # works
b = zeros(Float32,N);
laplace!(b,a); # works

using OffsetArrays: Origin
aPadded = Origin(0)(rand(Float32,N.+2));

using BenchmarkTools
@btime laplace!(b,a)       #1.168 ms (0 allocations: 0 bytes)
@btime laplace!(b,aPadded) #137.0 μs (0 allocations: 0 bytes) ! 8x faster !

using KernelAbstractions,CUDA,CUDA.CUDAKernels,Adapt
CUDA.allowscalar(false)

@kernel function lap_kernel(b, a)
    I = @index(Global, Cartesian)
    @fastmath @inbounds b[I] = a[I-δ(1)]+a[I+δ(1)]+a[I-δ(2)]+a[I+δ(2)]-4a[I]
end
backend(a) = CPU(); backend(a::CuArray) = CUDABackend()
laplaceKA!(b,a) = lap_kernel(backend(b), 64)(b, a, ndrange=size(b))
laplace!(b,a); # works

@btime laplaceKA!($b,$a)       #226.000 μs (200 allocations: 16.31 KiB)
@btime laplaceKA!($b,$aPadded) # 54.800 μs (200 allocations: 16.97 KiB) ! 4x faster !

b = adapt(CuArray,b);
a = adapt(CuArray,a);
aPadded = adapt(CuArray,aPadded);

@btime laplaceKA!($b,$a)       #2.167 μs (44 allocations: 2.12 KiB)
@btime laplaceKA!($b,$aPadded) #2.144 μs (47 allocations: 2.23 KiB)

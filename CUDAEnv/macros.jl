using WaterLily
using KernelAbstractions,CUDA,CUDA.CUDAKernels
CUDA.allowscalar(false)

struct Flow{Scal,Vect}
    u :: Vect
    σ :: Scal
    function Flow(N::NTuple{D}; f=identity, T = Float32) where D
        u = rand(T,N...,D)|>f
        σ = zeros(T,N)|>f
        new{typeof(σ),typeof(u)}(u,σ)
    end
end

@inline δ(i,I::CartesianIndex{m}) where{m} = CartesianIndex(ntuple(j -> j==i ? 1 : 0, m))
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@fastmath @inline function div_operator(I::CartesianIndex{m},u) where {m} 
    init=zero(eltype(u))
    for i in 1:m
     init += @inbounds ∂(i,I,u)
    end
    return init
end

@macroexpand1 @inside flow.σ[I] = div_operator(I,flow.u)
@macroexpand1 WaterLily.@loop flow.σ[I] = div_operator(I, flow.u) over I ∈ inside(flow.σ)

flow = Flow((6,6),f=cu)
WaterLily.@loop flow.σ[I] = div_operator(I, flow.u) over I ∈ CartesianIndices((2:4,1:2))
flow.σ |> Array
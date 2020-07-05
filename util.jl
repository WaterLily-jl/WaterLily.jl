@inline CI(a...) = CartesianIndex(a...)
@inline δ(a,d::Int) = CI(ntuple(i -> i==a ? 1 : 0, d))
@inline δ(a,I::CartesianIndex{N}) where {N} = δ(a,N)

@inline CR(a...) = CartesianIndices(a...)
@inline inside(M::NTuple{N,Int}) where {N} = CR(ntuple(i-> 2:M[i]-1,N))
@inline inside(a::Array; reverse::Bool=false) =
        reverse ? Iterators.reverse(inside(size(a))) : inside(size(a))
@inline inside_u(N::NTuple{n,T}) where {n,T} = CR(ntuple(i->2:N[i],n-1))

@fastmath function Σinside(a::Array{Float64},f=identity)
    s = 0.
    @simd for I ∈ inside(a)
        s += @inbounds f(a[I])
    end
    return s
end
@fastmath L₂(a::Array{Float64}) = Σinside(a,abs2)

@fastmath function median(a,b,c)
    x = a-b
    if x*(b-c) ≥ 0
        return b
    elseif x*(a-c) > 0
        return c
    else
        return a
    end
end

using Images,Plots
show(f) = plot(Gray.(f'[end:-1:1,:]))
show(f,fmin,fmax) = show((f.-fmin)/(fmax-fmin))
show_scaled(σ) = show(σ,minimum(σ),maximum(σ))

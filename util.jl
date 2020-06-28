@inline CR(a...) = CartesianIndices(a...)
@inline CI(a...) = CartesianIndex(a...)
@inline δ(a,d::Int) = CI(ntuple(i -> i==a ? 1 : 0, d))
@inline δ(a,I::CartesianIndex{N}) where {N} = δ(a,N)

@inline inside(M::NTuple{N,Int}) where {N} = CR(ntuple(i-> 2:M[i]-1,N))
@inline inside(a::Array) = inside(size(a))

using Images,Plots
show(f) = plot(Gray.(f'[end:-1:1,:]))
show(f,fmin,fmax) = show((f.-fmin)/(fmax-fmin))
show_scaled(σ) = show(σ,minimum(σ),maximum(σ))

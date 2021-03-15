@inline CI(a...) = CartesianIndex(a...)
@inline δ(a,d::Int) = CI(ntuple(i -> i==a ? 1 : 0, d))
@inline δ(a,I::CartesianIndex{N}) where {N} = δ(a,N)

@inline CR(a...) = CartesianIndices(a...)
@inline inside(M::NTuple{N,Int}) where {N} = CR(ntuple(i-> 2:M[i]-1,N))
@inline inside(a::Array; reverse::Bool=false) =
        reverse ? Iterators.reverse(inside(size(a))) : inside(size(a))
@inline inside_u(N::NTuple{n,T}) where {n,T} = CR(ntuple(i->2:N[i],n-1))

import Base.mapreduce
@fastmath function mapreduce(f,op,R::CartesianIndices;init=0.)
    val = init
    @inbounds @simd for I ∈ R
        val = op(val,f(I))
    end
    val
end
L₂(a::Array{Float64}) = mapreduce(I->@inbounds(abs2(a[I])),+,inside(a))

macro inside(ex)
    @assert ex.head==:(=)
    a,I = Meta.parse.(split(string(ex.args[1]),union("[","]")))
    return quote
        @inbounds @simd for $I ∈ inside($a)
            $ex
        end
    end |> esc
end

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

"""
    apply(f, N...)

Apply a vector function f(i,x) to the faces of a uniform staggered grid.
"""
function apply(f,N...)
    # TODO be more clever with the type
    c = Array{Float64}(undef,N...)
    apply!(f,c)
    return c
end
function apply!(f,c)
    N = size(c)
    for b ∈ 1:N[end]
        @simd for I ∈ CR(N[1:end-1])
            x = collect(Float16, I.I) # location at cell center
            x[b] -= 0.5               # location at face
            @inbounds c[I,b] = f(b,x) # apply function to location
        end
    end
end

function BC!(a::Array{T,4},A,f=1) where T
    for k∈1:size(a,3), j∈1:size(a,2)
        a[1,j,k,1] = a[2,j,k,1] = a[size(a,1),j,k,1] = f*A[1]
        a[1,j,k,2] = a[2,j,k,2]; a[size(a,1),j,k,2] = a[size(a,1)-1,j,k,2]
        a[1,j,k,3] = a[2,j,k,3]; a[size(a,1),j,k,3] = a[size(a,1)-1,j,k,3]
    end
    for k∈1:size(a,3), i∈1:size(a,1)
        a[i,1,k,2] = a[i,2,k,2] = a[i,size(a,2),k,2] = f*A[2]
        a[i,1,k,1] = a[i,2,k,1]; a[i,size(a,2),k,1] = a[i,size(a,2)-1,k,1]
        a[i,1,k,3] = a[i,2,k,3]; a[i,size(a,2),k,3] = a[i,size(a,2)-1,k,3]
    end
    for j∈1:size(a,2), i∈1:size(a,1)
        a[i,j,1,3] = a[i,j,2,3] = a[i,j,size(a,3),3] = f*A[3]
        a[i,j,1,1] = a[i,j,2,1]; a[i,j,size(a,3),1] = a[i,j,size(a,3)-1,1]
        a[i,j,1,2] = a[i,j,2,2]; a[i,j,size(a,3),2] = a[i,j,size(a,3)-1,2]
    end
end
function BC!(a::Array{T,3},A,f=1) where T
    for j∈1:size(a,2)
        a[1,j,1] = a[2,j,1] = a[size(a,1),j,1] = f*A[1]
        a[1,j,2] = a[2,j,2]; a[size(a,1),j,2] = a[size(a,1)-1,j,2]
    end
    for i∈1:size(a,1)
        a[i,1,2] = a[i,2,2] = a[i,size(a,2),2] = f*A[2]
        a[i,1,1] = a[i,2,1]; a[i,size(a,2),1] = a[i,size(a,2)-1,1]
    end
end
function BC!(a::Array{T,3}) where T
    for k∈1:size(a,3), j∈1:size(a,2)
        a[1,j,k] = a[2,j,k]; a[size(a,1),j,k] = a[size(a,1)-1,j,k]
    end
    for k∈1:size(a,3), i∈1:size(a,1)
        a[i,1,k] = a[i,2,k]; a[i,size(a,2),k] = a[i,size(a,2)-1,k]
    end
    for j∈1:size(a,2), i∈1:size(a,1)
        a[i,j,1] = a[i,j,2]; a[i,j,size(a,3)] = a[i,j,size(a,3)-1]
    end
end
function BC!(a::Array{T,2}) where T
    for j∈1:size(a,2)
        a[1,j] = a[2,j]; a[size(a,1),j] = a[size(a,1)-1,j]
    end
    for i∈1:size(a,1)
        a[i,1] = a[i,2]; a[i,size(a,2)] = a[i,size(a,2)-1]
    end
end

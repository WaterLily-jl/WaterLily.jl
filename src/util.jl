@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
@inline δ(i,N::Int) = CI(ntuple(j -> j==i ? 1 : 0, N))
@inline δ(i,I::CartesianIndex{N}) where {N} = δ(i,N)

"""
    inside(dims)
    inside(a) = inside(size(a))

Return CartesianIndices range excluding the ghost-cells on the boundaries of
a _scalar_ array `a` with `dims=size(a)`.
"""
@inline inside(dims::NTuple{N}) where {N} = CartesianIndices(ntuple(i-> 2:dims[i]-1,N))
@inline inside(a::AbstractArray) = inside(size(a))

"""
    inside_u(dims,j)

Return CartesianIndices range excluding the ghost-cells on the boundaries of
a _vector_ array on face `j` with size `dims`.
"""
function inside_u(dims::NTuple{N},j) where {N}
    CartesianIndices(ntuple( i-> i==j ? (3:dims[i]-1) : (2:dims[i]), N))
end
splitn(n) = Base.front(n),n[end]
size_u(u) = splitn(size(u))

"""
    L₂(a)

L₂ norm of array `a` excluding ghosts.
"""
L₂(a) = mapreduce(I->@inbounds(abs2(a[I])),+,inside(a))

"""
    @inside <expr>

Simple macro to automate efficient loops over cells excluding ghosts. For example

    @inside p[I] = sum(I.I)

becomes

    @inbounds @simd for I ∈ inside(p)
        p[I] = sum(I.I)
    end
"""
macro inside(ex)
    a,I = Meta.parse.(split(string(ex.args[1]),union("[",",","]")))
    return quote 
        WaterLily.@loop $ex over $I ∈ inside($a)
    end |> esc
end
macro loop(args...)
    ex,_,itr = args
    op,I,R = itr.args
    @assert op ∈ (:(∈),:(in))
    return quote
        @inbounds @simd for $I ∈ $R
            $ex
        end
    end |> esc
end

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

using StaticArrays
"""
    loc(i,I)

Location in space of the cell at CartesianIndex `I` at face `i`.
Using `i=0` returns the cell center s.t. `loc = I`.
"""
@inline loc(i,I) = SVector(I.I .- 0.5 .* δ(i,I).I)

"""
    apply!(f, c)

Apply a vector function `f(i,x)` to the faces of a uniform staggered array `c`.
"""
function apply!(f,c)
    N,n = size_u(c)
    for i ∈ 1:n
        @loop c[I,i] = f(i,loc(i,I)) over I ∈ CartesianIndices(N)
    end
end

"""
    slice(dims,i,j,low=1)

Return `CartesianIndices` range slicing through an array of size `dims` in
dimension `j` at index `i`. `low` optionally sets the lower extent of the range
in the other dimensions.
"""
function slice(dims::NTuple{N},i,j,low=1) where N
    CartesianIndices(ntuple( k-> k==j ? (i:i) : (low:dims[k]), N))
end

"""
    BC!(a,A,f=1)

Apply boundary conditions to the ghost cells of a _vector_ field. A Dirichlet
condition `a[I,i]=f*A[i]` is applied to the vector component _normal_ to the domain
boundary. For example `aₓ(x)=f*Aₓ ∀ x ∈ minmax(X)`. A zero Nuemann condition
is applied to the tangential components.
"""
function BC!(a,A,f=1)
    N,n = size_u(a)
    for j ∈ 1:n, i ∈ 1:n
        if i==j # Normal direction, Dirichlet
            for s ∈ (1,2,N[j])
                @loop a[I,i] = f*A[i] over I ∈ slice(N,s,j)
            end
        else    # Tangential directions, Neumann
            @loop a[I,i] = a[I+δ(j,I),i] over I ∈ slice(N,1,j)
            @loop a[I,i] = a[I-δ(j,I),i] over I ∈ slice(N,N[j],j)
        end
    end
end

"""
    BC!(a)

Apply zero Nuemann boundary conditions to the ghost cells of a _scalar_ field.
"""
function BC!(a)
    N = size(a)
    for j ∈ 1:length(N)
        @loop a[I] = a[I+δ(j,I)] over I ∈ slice(N,1,j)
        @loop a[I] = a[I-δ(j,I)] over I ∈ slice(N,N[j],j)
    end
end

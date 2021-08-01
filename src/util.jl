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

import Base.mapreduce
"""
    mapreduce(f,op,R::CartesianIndices;init=0.)

Apply a function `f(I:CartesianIndex)` and redution operation `op` over a
CartesianIndices range `R`. Optionally specific the initial value `init`
to the reduction.
"""
@fastmath function mapreduce(f,op,R::CartesianIndices;init=0.)
    val = init
    @inbounds @simd for I ∈ R
        val = op(val,f(I))
    end
    val
end
"""
    L₂(a)

L₂ norm of array `a` excluding ghosts.
"""
L₂(a) = mapreduce(I->@inbounds(abs2(a[I])),+,inside(a))

"""
    @inside

Simple macro to automate efficient loops over cells excluding ghosts. For example

    @inside p[I] = loc(0,I)

will generate the code

    @inbounds @simd for I ∈ inside(p)
        p[I] = loc(0,I)
    end

Note: Someone better at meta-programming could help generalize this to work for
other cases such as `@inside p[I] += f(I)` or `@inside u[I,j] = f(j,I)` etc.
"""
macro inside(ex)
    @assert ex.head==:(=)
    a,I = Meta.parse.(split(string(ex.args[1]),union("[","]")))
    return quote
        @inbounds @simd for $I ∈ inside($a)
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
        @inbounds @simd for I ∈ CartesianIndices(N)
            c[I,i] = f(i,loc(i,I))
        end
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
        if i==j # Inline direction
            for s ∈ (1,2,N[j]); @simd for I ∈ slice(N,s,j)
                a[I,i] = f*A[i] # Dirichlet
            end; end
        else    # Perpendicular directions
            @simd for I ∈ slice(N,1,j)
                a[I,i] = a[I+δ(j,I),i] # Neumann
            end
            @simd for I ∈ slice(N,N[j],j)
                a[I,i] = a[I-δ(j,I),i] # Neumann
            end
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
        @simd for I ∈ slice(N,1,j)
            a[I] = a[I+δ(j,I)] # Neumann
        end
        @simd for I ∈ slice(N,N[j],j)
            a[I] = a[I-δ(j,I)] # Neumann
        end
    end
end

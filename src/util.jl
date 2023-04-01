@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
@inline δ(i,N::Int) = CI(ntuple(j -> j==i ? 1 : 0, N))
@inline δ(i,I::CartesianIndex{N}) where {N} = δ(i,N)

# """
#     O(D=0, d=1)

# Returns the Offset.Origin function for a D-dimensional array.
# O() returns Origin(0).
# O(D) returns the origin for a vector field array: O(2) = Origin(0, 0, 1), where the
# last dimension has no offset since it refers to the vector dimensions.
# O(D, d) same as O(D) but with repeated ones in the last dimensions: O(2, 2) = Origin(0, 0, 1, 1).
# """
# O(D=0, d=1) = OffsetArrays.Origin(D > 0 ? (zeros(Int, D)..., ones(Int, d)...) : 0)

# """
#     ArrayT

# Alias for CPU Array or GPU CuArray depending on the backend.
# """
# ArrayT = (backend == CPU()) ? Array : CuArray

"""
    inside(dims)
    inside(a) = inside(size(a))

Return CartesianIndices range excluding the ghost-cells on the boundaries of
a _scalar_ array `a` with `dims=size(a)`.
"""
@inline inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))

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
L₂(a) = sum(@inbounds(abs2(a[I])) for I ∈ inside(a))

"""
    @inside <expr>

Simple macro to automate efficient loops over cells excluding ghosts. For example

    @inside p[I] = sum(I.I)

becomes

    @loop p[I] = sum(I.I) over I ∈ inside(p)

See `inside` and `@loop`.
"""
macro inside(ex)
    a,I = Meta.parse.(split(string(ex.args[1]),union("[",",","]")))
    return quote
        WaterLily.@loop $ex over $I ∈ inside($a)
    end |> esc
end

"""
    @loop <expr> over I ∈ R

Simple macro to automate efficient loops. For example

    @loop r[I] += sum(I.I) over I ∈ CartesianIndex(r)

becomes

    @inbounds Polyester.@batch for I ∈ CartesianIndex(r)
        r[I] += sum(I.I)
    end

using package Polyester to apply loop vectorization and multithreading.
"""
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

_ENABLE_PUSH = true
DISABLE_PUSH() = (global _ENABLE_PUSH = false)
ENABLE_PUSH() = (global _ENABLE_PUSH = true)

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
    slice(dims,i,j,low=1,trim=0)

Return `CartesianIndices` range slicing through an array of size `dims` in
dimension `j` at index `i`. `low` optionally sets the lower extent of the range
in the other dimensions. `trim` removes elements from the back indices.
"""
function slice(dims::NTuple{N}, i, j, low = 1, trim = 0) where N
    CartesianIndices(ntuple(k-> k == j ? (i:i) : (low:dims[k] - trim), N))
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
# function BC!(u, U, bc, f = 1.0)
#     _BC!(backend, 64)(u, U, bc, f, ndrange=size(bc))
# end
# @kernel function _BC!(u, @Const(U), @Const(bc), @Const(f))
#     i = @index(Global, Linear)
#     ghostI, donorI, di = bc[i][1], bc[i][2], bc[i][3]
#     _, D = size_u(u)
#     for d ∈ 1:D
#         if d == di
#             u[ghostI, d] = f * U[d]
#         else
#             u[ghostI, d] = u[donorI, d]
#         end
#     end
# end

"""
    BC!(a)

Apply zero Nuemann boundary conditions to the ghost cells of a _scalar_ field.
"""
function BC!(a)
    N = size(a)
    for j ∈ eachindex(N)
        @loop a[I] = a[I+δ(j,I)] over I ∈ slice(N,1,j)
        @loop a[I] = a[I-δ(j,I)] over I ∈ slice(N,N[j],j)
    end
end
# function BC!(u, bc)
#     _BC!(backend, 64)(u, bc, ndrange=size(bc))
# end
# @kernel function _BC!(u, @Const(bc))
#     i = @index(Global, Linear)
#     ghostI, donorI = bc[i][1], bc[i][2]
#     u[ghostI] = u[donorI]
# end
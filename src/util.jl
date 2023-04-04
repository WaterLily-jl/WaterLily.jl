using KernelAbstractions, Adapt, CUDA, CUDA.CUDAKernels

if Base.find_package("CUDA") !== nothing
    using CUDA.CUDAKernels
    const backend = CUDABackend()
    CUDA.allowscalar(false)
else
    const backend = CPU()
end

@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
@inline δ(i,N::Int) = CI(ntuple(j -> j==i ? 1 : 0, N))
@inline δ(i,I::CartesianIndex{N}) where {N} = δ(i,N)

using OffsetArrays
"""
    OA(D=0, d=1)

Returns the Offset.Origin function for a D-dimensional array.
OA() applies Origin(0), shifting the OffsetArray axes to start from 0.
OA(D≠0,d) shifts the first D axes to start from 0 and keeps the last d axes starting from 1.
"""
OA(D=0, d=1) = OffsetArrays.Origin(D > 0 ? (zeros(Int, D)..., ones(Int, d)...) : 0)

"""
    inside(a)

Return CartesianIndices range excluding the a single cell of ghosts on all boundaries.
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
    adapt!(u)
Adapt an array `u` to a CPU or CUDA `backend`
"""
adapt!(u) = backend == CPU() ? adapt(Array, u) : adapt(CuArray, u) # outer scope (general) backend
adapt!(u, b) = b == CPU() ? adapt(Array, u) : adapt(CuArray, u)

# """
#     ArrayT

# Alias for CPU Array or GPU CuArray depending on the backend.
# """
# ArrayT = (backend == CPU()) ? Array : CuArray
#
# """
#     @inside <expr>

# Simple macro to automate efficient loops over cells excluding ghosts. For example

#     @inside p[I] = sum(I.I)

# becomes

#     @loop p[I] = sum(I.I) over I ∈ size(p).-2

# See `@loop`.
# """
# macro inside(ex)
#     # Make sure its a single assignment
#     @assert ex.head == :(=) && ex.args[1].head == :(ref)
#     a,I = ex.args[1].args[1:2]
#     return quote # loop over the size of the reference
#         @loop $ex over $I ∈ size($a).-2
#     end |> esc
# end

# """
#     @loop <expr> over I ∈ ndrange

# Simple macro to automate kernel. For example

#     @loop r[I] += sum(I.I) over I ∈ size(r).-2

# becomes

#     @kernel function f(r)
#         I ∈ @index(Global,Cartesian)
#         r[I] += sum(I.I)
#     end
#     f(backend(r),64)(r,ndrange=size(r).-2)

# using package KernelAbstractions to run on CPUs or GPUs.
# """
# macro loop(args...)
#     ex,_,itr = args
#     _,I,R = itr.args; sym = []
#     grab!(sym,ex)     # get arguments and replace composites in `ex`
#     setdiff!(sym,[I]) # don't want to pass I as an argument
#     return quote
#         @kernel function f($(rep.(sym)...)) # replace composite arguments
#             $I = @index(Global, Cartesian)
#             $ex
#         end
#         f(KernelAbstractions.get_backend($(sym[1])),64)($(sym...),ndrange=$R)
#         return nothing
#     end |> esc
# end
# function grab!(sym,ex::Expr)
#     ex.head == :. && return union!(sym,[ex])    # keep composited names without recursion
#     start = ex.head==:(call) ? 2 : 1            # don't grab function names
#     foreach(a->grab!(sym,a),ex.args[start:end]) # recurse
#     ex.args .= rep.(ex.args)                    # replace composite names with value
# end
# grab!(sym,ex::Symbol) = union!(sym,[ex])        # keep symbol names
# grab!(sym,ex) = nothing
# rep(ex) = ex
# rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[2].value) : ex

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

# """
#     apply!(f, c)

# Apply a vector function `f(i,x)` to the faces of a uniform staggered array `c`.
# """
# function apply!(f,c)
#     N,n = size_u(c)
#     for i ∈ 1:n
#         @loop c[I,i] = f(i,loc(i,I)) over I ∈ CartesianIndices(N)
#     end
# end
# function apply!(c, f) # swapped arguments since the exclamation mark is supposed to modify the front arguments
#     _apply!(KernelAbstractions.get_backend(c), 64)(c, f, ndrange=Base.front(size(c)))
# end
# @kernel function _apply!(c, @Const(f))
#     I = @index(Global, Cartesian)
#     _, D = size_u(c)
#     for d ∈ 1:D
#         c[I, d] = f(d, loc(d, I))
#     end
# end

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
    bc_indices(Ng)

Given an array size Ng = (N, M, ...), that includes the ghost cells, it returns a
Vector of Tuple(s) in which each Tuple is composed of a ghost cell CartesianIndex,
its respective donor cell CartesianIndex, and the normal direction between them:
[Tuple{CartesianIndex, CartesianIndex, Int}, ...]
"""
function bc_indices(Ng)
    D = length(Ng)
    bc_list = Tuple{CartesianIndex, CartesianIndex, Int}[]
    for d ∈ 1:D
        slice_ghost_start = slice(Ng, 0, d, 1, 2)
        slice_donor_start = slice_ghost_start .+ δ(d, D)
        slice_ghost_end = slice(Ng, Ng[d] - 1, d, 1, 2)
        slice_donor_end = slice_ghost_end .- δ(d, D)
        push!(bc_list, zip(slice_ghost_start, slice_donor_start, ntuple(x -> d, length(slice_ghost_start)))...,
            zip(slice_ghost_end, slice_donor_end, ntuple(x -> d, length(slice_ghost_end)))...)
    end
    return Tuple.(bc_list)
end

"""
    BC!(a,A,f=1)

Apply boundary conditions to the ghost cells of a _vector_ field. A Dirichlet
condition `a[I,i]=f*A[i]` is applied to the vector component _normal_ to the domain
boundary. For example `aₓ(x)=f*Aₓ ∀ x ∈ minmax(X)`. A zero Nuemann condition
is applied to the tangential components.
"""
# function BC!(a,A,f=1)
#     N,n = size_u(a)
#     for j ∈ 1:n, i ∈ 1:n
#         if i==j # Normal direction, Dirichlet
#             for s ∈ (1,2,N[j])
#                 @loop a[I,i] = f*A[i] over I ∈ slice(N,s,j)
#             end
#         else    # Tangential directions, Neumann
#             @loop a[I,i] = a[I+δ(j,I),i] over I ∈ slice(N,1,j)
#             @loop a[I,i] = a[I-δ(j,I),i] over I ∈ slice(N,N[j],j)
#         end
#     end
# end
function BC!(u, U, bc, f = 1.0)
    _BC!(KernelAbstractions.get_backend(u), 64)(u, U, bc, f, ndrange=size(bc))
end
@kernel function _BC!(u, @Const(U), @Const(bc), @Const(f))
    i = @index(Global, Linear)
    ghostI, donorI, di = bc[i][1], bc[i][2], bc[i][3]
    _, D = size_u(u)
    for d ∈ 1:D
        if d == di
            u[ghostI, d] = f * U[d]
        else
            u[ghostI, d] = u[donorI, d]
        end
    end
end

"""
    BC!(a)

Apply zero Nuemann boundary conditions to the ghost cells of a _scalar_ field.
"""
# function BC!(a)
#     N = size(a)
#     for j ∈ eachindex(N)
#         @loop a[I] = a[I+δ(j,I)] over I ∈ slice(N,1,j)
#         @loop a[I] = a[I-δ(j,I)] over I ∈ slice(N,N[j],j)
#     end
# end
function BC!(u, bc)
    _BC!(KernelAbstractions.get_backend(u), 64)(u, bc, ndrange=size(bc))
end
@kernel function _BC!(u, @Const(bc))
    i = @index(Global, Linear)
    ghostI, donorI = bc[i][1], bc[i][2]
    u[ghostI] = u[donorI]
end
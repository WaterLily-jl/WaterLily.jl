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

# """
#     inside_u(dims,j)

# Return CartesianIndices range excluding the ghost-cells on the boundaries of
# a _vector_ array on face `j` with size `dims`.
# """
# function inside_u(dims::NTuple{N},j) where {N}
    # CartesianIndices(ntuple( i-> i==j ? (3:dims[i]-1) : (2:dims[i]), N))
# end
splitn(n) = Base.front(n),last(n)
size_u(u) = splitn(size(u))

"""
    L₂(a)

L₂ norm of array `a` excluding ghosts.
"""
L₂(a) = mapreduce(abs2,+,@inbounds a[inside(a)])
L₂(a::OffsetArray) = L₂(parent(a))
"""
    @inside <expr>

Simple macro to automate efficient loops over cells excluding ghosts. For example

    @inside p[I] = sum(I.I)

becomes

    @loop p[I] = sum(I.I) over I ∈ size(p).-2

See `@loop`.
"""
macro inside(ex)
    # Make sure its a single assignment
    @assert ex.head == :(=) && ex.args[1].head == :(ref)
    a,I = ex.args[1].args[1:2]
    return quote # loop over the size of the reference
        WaterLily.@loop $ex over $I ∈ size($a).-2
    end |> esc
end

using KernelAbstractions,CUDA,CUDA.CUDAKernels
using KernelAbstractions: get_backend
"""
    @loop <expr> over I ∈ ndrange

Simple macro to automate kernel. For example

    @loop r[I] += sum(I.I) over I ∈ size(r).-2

becomes

    @kernel function f(r)
        I ∈ @index(Global,Cartesian)
        r[I] += sum(I.I)
    end
    f(backend(r),64)(r,ndrange=size(r).-2)

using package KernelAbstractions to run on CPUs or GPUs.
"""
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args; sym = []
    grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    @gensym kern
    return quote
        @kernel function $kern($(rep.(sym)...)) # replace composite arguments
            $I = @index(Global, Cartesian)
            $ex
        end
        $kern(get_backend($(sym[1])),64)($(sym...),ndrange=$R)
    end |> esc
end
function grab!(sym,ex::Expr)
    ex.head == :. && return union!(sym,[ex])      # grab composite name and return
    start = ex.head==:(call) ? 2 : 1              # don't grab function names
    foreach(a->grab!(sym,a),ex.args[start:end])   # recurse into args
    ex.args[start:end] = rep.(ex.args[start:end]) # replace composites in args
end
grab!(sym,ex::Symbol) = union!(sym,[ex])        # grab symbol name
grab!(sym,ex) = nothing
rep(ex) = ex
rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[2].value) : ex

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
        @loop c[I,i] = f(i,loc(i,I)) over I ∈ N .- 2
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
        slice_ghost_start = slice(Ng, 0, d, 0, 1)
        slice_donor_start = slice_ghost_start .+ δ(d, D)
        slice_ghost_end = slice(Ng, Ng[d] - 1, d, 0, 1)
        slice_donor_end = slice_ghost_end .- δ(d, D)
        push!(bc_list,
            zip(slice_ghost_start, slice_donor_start, ntuple(x -> d, length(slice_ghost_start)))...,
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
function BC!(u, U, bc, f = 1.0)
    _BC!(get_backend(u), 64)(u, U, bc, f, ndrange=size(bc))
end
@kernel function _BC!(u, @Const(U), @Const(bc), @Const(f))
    i = @index(Global, Linear)
    ghostI, donorI, di = bc[i][1], bc[i][2], bc[i][3]
    D = length(U)
    for d ∈ 1:D
        if d == di
            u[ghostI, d] = f * U[d]
            if ghostI[d] == 0
                u[ghostI + δ(d, D), d] = f * U[d]
            end
        elseif ghostI[d] > 1
            u[ghostI, d] = u[donorI, d]
        end
    end
end

"""
    BC!(a)

Apply zero Nuemann boundary conditions to the ghost cells of a _scalar_ field.
"""
function BC!(u, bc)
    _BC!(get_backend(u), 64)(u, bc, ndrange=size(bc))
end
@kernel function _BC!(u, @Const(bc))
    i = @index(Global, Linear)
    ghostI, donorI = bc[i][1], bc[i][2]
    u[ghostI] = u[donorI]
end
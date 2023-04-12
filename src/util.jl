@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
@inline δ(i,N::Int) = CI(ntuple(j -> j==i ? 1 : 0, N))
@inline δ(i,I::CartesianIndex{N}) where {N} = δ(i,N)

"""
    inside(a)

Return CartesianIndices range excluding a single layer of cells on all boundaries.
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

using CUDA
"""
    L₂(a)

L₂ norm of array `a` excluding ghosts.
"""
L₂(a) = sum(abs2,@inbounds(a[I]) for I ∈ inside(a))
L₂(a::CuArray,R::CartesianIndices=inside(a)) = mapreduce(abs2,+,@inbounds(a[R]))
"""
    @inside <expr>

Simple macro to automate efficient loops over cells excluding ghosts. For example

    @inside p[I] = sum(I.I)

becomes

    @loop p[I] = sum(I.I) over I ∈ inside(p)

See `@loop`.
"""
macro inside(ex)
    # Make sure its a single assignment
    @assert ex.head == :(=) && ex.args[1].head == :(ref)
    a,I = ex.args[1].args[1:2]
    return quote # loop over the size of the reference
        WaterLily.@loop $ex over $I ∈ inside($a)
    end |> esc
end

using KernelAbstractions,CUDA,CUDA.CUDAKernels
using KernelAbstractions: get_backend
"""
    @loop <expr> over I ∈ ndrange

Simple macro to automate kernel. For example

    @loop a[I] += sum(I.I) over I ∈ R

becomes     

    @kernel function f(a)
        I ∈ @index(Global,Cartesian)+R[1]-oneunit(R[1])
        a[I] += sum(I.I)
    end
    f(backend(a),64)(a,ndrange=size(R))

using package KernelAbstractions to run on CPUs or GPUs.
"""
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args; sym = []
    grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    @gensym kern
    return quote
        @kernel function $kern($(rep.(sym)...),@Const(I0)) # replace composite arguments
            $I = @index(Global,Cartesian)
            $I += I0
            $ex
        end
        $kern(get_backend($(sym[1])),64)($(sym...),$R[1]-oneunit($R[1]),ndrange=size($R))
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
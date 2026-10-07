using KernelAbstractions: get_backend, @index, @kernel, supports_float64
using LoggingExtras

# custom log macro
_psolver = Logging.LogLevel(-123) # custom log level for pressure solver, needs the negative sign
macro log(exs...)
    quote
        @logmsg _psolver $(map(x -> esc(x), exs)...)
    end
end
"""
    logger(fname="WaterLily")

Set up a logger to write the pressure solver data to a logging file named `WaterLily.log`.
"""
function logger(fname::String="WaterLily")
    ENV["JULIA_DEBUG"] = all
    logger = FormatLogger(ifelse(fname[end-3:end]==".log",fname[1:end-4],fname)*".log"; append=false) do io, args
        args.level == _psolver && print(io, args.message)
    end;
    global_logger(logger);
    # put header in file
    @log "p/c, iter, r∞, r₁, ω\n"
end

@inline CI(a...) = CartesianIndex(a...)
"""
    CIj(j,I,k)
Replace jᵗʰ component of CartesianIndex with k
"""
CIj(j,I::CartesianIndex{d},k) where d = CI(ntuple(i -> i==j ? k : I[i], d))

"""
    δ(i,::Val{N})
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

"""
    inside(a;buff=1)

Return CartesianIndices range excluding a single layer of cells on all boundaries.
"""
@inline inside(a::AbstractArray;buff=1) = CartesianIndices(map(ax->first(ax)+buff:last(ax)-buff,axes(a)))

"""
    inside_u(dims,j)

Return CartesianIndices range excluding the ghost-cells on the boundaries of
a _vector_ array on face `j` with size `dims`.
"""
function inside_u(dims::NTuple{N},j) where {N}
    CartesianIndices(ntuple( i-> i==j ? (3:dims[i]-1) : (2:dims[i]), N))
end
@inline inside_u(dims::NTuple{N}) where N = CartesianIndices((map(i->(2:i-1),dims)...,1:N))
@inline inside_u(u::AbstractArray) = CartesianIndices(map(i->(2:i-1),size(u)[1:end-1]))
splitn(n) = Base.front(n),last(n)
size_u(u) = splitn(size(u))

"""
    @inside <expr>

Simple macro to automate efficient loops over cells excluding ghosts. For example,

    @inside p[I] = sum(loc(0,I))

becomes

    @loop p[I] = sum(loc(0,I)) over I ∈ inside(p)

See [`@loop`](@ref).
"""
macro inside(ex)
    # Make sure it's a single assignment
    @assert ex.head == :(=) && ex.args[1].head == :(ref)
    a,I = ex.args[1].args[1:2]
    return quote # loop over the size of the reference
        WaterLily.@loop $ex over $I ∈ inside($a)
    end |> esc
end

# Could also use ScopedValues in Julia 1.11+
using Preferences
const backend = @load_preference("backend", "KernelAbstractions")
function set_backend(new_backend::String)
    if !(new_backend in ("SIMD", "KernelAbstractions"))
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend" => new_backend)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

"""
    @loop <expr> over <I ∈ R>

Macro to automate fast loops using @simd when running in serial,
or KernelAbstractions when running multi-threaded CPU or GPU.

For example

    @loop a[I,i] += sum(loc(i,I)) over I ∈ R

becomes

    @inbounds @simd for J ∈ CartesianIndices(R)
        I = R[J]
        @fastmath @inbounds a[I,i] += sum(loc(i,I))
    end

on serial execution, or

    @kernel function kern(a,i,R)
        I = R[@index(Global,Cartesian)]
        @fastmath @inbounds a[I,i] += sum(loc(i,I))
    end
    kern(get_backend(a),64)(a,i,R,ndrange=size(R))

when multi-threading on CPU or using CuArrays.
Note that `get_backend` is used on the _first_ variable in `expr` (`a` in this example).

`R` is any array, such as a `CartesianIndices`, a [`face`](@ref) or a vector on the same device as `a`,
and `I` takes its values.
"""
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args
    sym = []
    grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    symT = [gensym() for _ in 1:length(sym)] # generate a list of types for each symbol
    symWtypes = joinsymtype(rep.(sym),symT) # symbols with types: [a::A, b::B, ...]
    @gensym(kern, kern_, R_, J_) # generate unique kernel function names for serial and KA execution, and the range and launch index
    @static if backend == "KernelAbstractions"
        return quote
            @kernel function $kern_($(symWtypes...),$R_) where {$(symT...)} # replace composite arguments
                $J_ = @index(Global,Cartesian)
                $I = @inbounds $R_[$J_] # the range gives the cell index
                @fastmath @inbounds $ex
            end
            function $kern($kern_,$(symWtypes...)) where {$(symT...)} # kernel passed as argument: capturing it would box it
                $R_ = $R
                $kern_(get_backend($(rep(sym[1]))),64)($(rep.(sym)...),$R_,ndrange=size($R_))
            end
            $kern($kern_,$(sym...))
        end |> esc
    else # backend == "SIMD"
        return quote
            function $kern($(symWtypes...)) where {$(symT...)}
                $R_ = $R
                @inbounds @simd for $J_ ∈ CartesianIndices($R_) # @inbounds need for @simd vectorization, CartesianIndices for nested loops over any range
                    $I = $R_[$J_]
                    @fastmath @inbounds $ex
                end
            end
            $kern($(sym...))
        end |> esc
    end
end
function grab!(sym,ex::Expr)
    ex.head == :. && return union!(sym,[ex])      # grab composite name and return
    start = ex.head==:(call) ? 2 : 1              # don't grab function names
    foreach(a->grab!(sym,a),ex.args[start:end])   # recurse into args
    ex.args[start:end] = rep.(ex.args[start:end]) # replace composites in args
end
grab!(sym,ex::Symbol) = union!(sym,[ex])          # grab symbol name
grab!(sym,ex) = nothing
rep(ex) = ex
rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[2].value) : ex
joinsymtype(sym::Symbol,symT::Symbol) = Expr(:(::), sym, symT)
joinsymtype(sym,symT) = zip(sym,symT) .|> x->joinsymtype(x...)

using StaticArrays
"""
    loc(i,I,T) = loc(Ii,T)

Location in space of the cell at CartesianIndex `I` at face `i`.
Using `i=0` returns the cell center s.t. `loc = I`. `T` sets the return number type.
"""
@inline loc(i,I::CartesianIndex{N},T=Float32) where N = SVector{N,T}(I.I .- T(1.5) .- δ(i,I).I ./T(2))
@inline loc(Ii::CartesianIndex,T=Float32) = loc(last(Ii),Base.front(Ii),T)
Base.last(I::CartesianIndex) = last(I.I)
Base.front(I::CartesianIndex) = CI(Base.front(I.I))
"""
    slice(dims,i,j,low=1)

Return `CartesianIndices` range slicing through an array of size `dims` in
dimension `j` at index `i`. `low` optionally sets the lower extent of the range
in the other dimensions.
"""
function slice(dims::NTuple{N},i,j,low=1) where N
    CartesianIndices(ntuple( k-> k==j ? (i:i) : (low:dims[k]), N))
end

# `R` with `f` applied on indexing, as MappedArrays.mappedarray (renamed to not clash with it)
struct MappedArr{T,N,F,C<:AbstractArray{<:Any,N}} <: AbstractArray{T,N}
    f::F; R::C
end
MappedArr(f,R) = MappedArr{Base.promote_op(f,eltype(R)),ndims(R),typeof(f),typeof(R)}(f,R)
Base.size(M::MappedArr) = size(M.R)
Base.@propagate_inbounds Base.getindex(M::MappedArr{T,N},J::Vararg{Int,N}) where {T,N} = M.f(M.R[J...])
insert(J::CartesianIndex{M},i,j) where M = CI(ntuple(k -> k<j ? J[k] : k==j ? i : J[k-1], M+1)) # put index i back in dimension j
"""
    face(R,j)
    face(dims,i,j,low=1) = face(slice(dims,i,j,low),j)

The cells of a range `R` one cell thick in dimension `j` (such as a `slice`), as a `MappedArr` over its
other dimensions. `@loop` launches over those, so the 64-thread workgroups lie along the face, also when
it is normal to `j=1`.
"""
face(R::CartesianIndices{N},j) where N = (i=only(R.indices[j]); MappedArr(J->insert(J,i,j),CartesianIndices(ntuple(k -> R.indices[k<j ? k : k+1], N-1))))
face(dims::NTuple,i,j,low=1) = face(slice(dims,i,j,low),j)

"""
    BC!(a,U,saveexit=false,perdir=(),t=0)

Apply boundary conditions to the ghost cells of a _vector_ field. A Dirichlet
condition `a[I,i]=U[i]` is applied to the vector component _normal_ to the domain
boundary. For example `aₓ(x)=Aₓ ∀ x ∈ minmax(X)`.
A zero Neumann condition is applied to the tangential components.
`saveexit=true` leaves the normal component on the `x` exit face (upper `x` boundary) untouched,
so a convective outflow set by `exitBC!` is kept and every other face is always overwritten.
`perdir` lists the dimensions using periodic BCs instead.
`t` is passed to `U` when it is a function `(i,x,t)->...`.
"""
BC!(a,U,saveexit=false,perdir=(),t=0) = BC!(a,(i,x,t)->U[i],saveexit,perdir,t)
function BC!(a,uBC::Function,saveexit=false,perdir=(),t=0)
    N,n = size_u(a)
    for i ∈ 1:n, j ∈ 1:n
        if j in perdir
            @loop a[I,i] = a[CIj(j,I,N[j]-1),i] over I ∈ face(N,1,j)
            @loop a[I,i] = a[CIj(j,I,2),i] over I ∈ face(N,N[j],j)
        else
            if i==j # Normal direction, Dirichlet
                for s ∈ (1,2)
                    @loop a[I,i] = uBC(i,loc(i,I,eltype(a)),t) over I ∈ face(N,s,j)
                end
                (!saveexit || i>1) && (@loop a[I,i] = uBC(i,loc(i,I,eltype(a)),t) over I ∈ face(N,N[j],j)) # overwrite exit
            else    # Tangential directions, Neumann
                @loop a[I,i] = uBC(i,loc(i,I,eltype(a)),t)+a[I+δ(j,I),i]-uBC(i,loc(i,I+δ(j,I),eltype(a)),t) over I ∈ face(N,1,j)
                @loop a[I,i] = uBC(i,loc(i,I,eltype(a)),t)+a[I-δ(j,I),i]-uBC(i,loc(i,I-δ(j,I),eltype(a)),t) over I ∈ face(N,N[j],j)
            end
        end
    end
end

"""
    exitBC!(u,u⁰,Δt)

Apply a 1D convection scheme to fill the ghost cell on the exit of the domain.
"""
function exitBC!(u,u⁰,Δt)
    N,_ = size_u(u)
    exitR = slice(N.-1,N[1],1,2)              # exit slice excluding ghosts
    U = sum(@view(u[slice(N.-1,2,1,2),1]))/length(exitR) # inflow mass flux
    @loop u[I,1] = u⁰[I,1]-U*Δt*(u⁰[I,1]-u⁰[I-δ(1,I),1]) over I ∈ face(exitR,1)
    ∮u = sum(@view(u[exitR,1]))/length(exitR)-U   # mass flux imbalance
    @loop u[I,1] -= ∮u over I ∈ face(exitR,1)     # correct flux
end
"""
    perBC!(a,perdir)

Apply periodic conditions to the ghost cells of a _scalar_ field.
"""
perBC!(a,::Tuple{}) = nothing
perBC!(a, perdir, N = size(a)) = for j ∈ perdir
    @loop a[I] = a[CIj(j,I,N[j]-1)] over I ∈ face(N,1,j)
    @loop a[I] = a[CIj(j,I,2)] over I ∈ face(N,N[j],j)
end

using ForwardDiff
using ForwardDiff: Dual, partials, Tag

# Inner-derivative tag for measure's gradient/jacobian/derivative. `≺` is
# overloaded so it always ranks newer than any `ForwardDiff.Tag`, folding the
# precedence comparison at compile time and sidestepping `tagcount` (order-
# sensitive on GPU codegen, the original cause of nested-FD crashes in kernels).
struct _InnerTag end
@inline ForwardDiff.:≺(::Type{<:Tag}, ::Type{_InnerTag}) = true
@inline ForwardDiff.:≺(::Type{_InnerTag}, ::Type{<:Tag}) = false
@inline ForwardDiff.:≺(::Type{_InnerTag}, ::Type{_InnerTag}) = false

# Tag-aware partial extractor. The fallback returns zero when `y` is not an
# `_InnerTag` dual — `f` did not depend on the seeded input so the inner
# derivative is exactly zero. Without it, an outer-tag `Dual` (from closure
# capture) would silently leak its outer partial.
@inline _ip(y::Dual{_InnerTag}, i::Int) = partials(y, i)
@inline _ip(y, ::Int) = zero(y)

# GPU-safe gradient/jacobian/derivative: seed `Dual{_InnerTag}` and extract
# `partials` directly, bypassing `extract_jacobian`/`valtype`. SVector inputs
# take the GPU-safe path; other inputs (plain `AbstractVector`, e.g. unit tests)
# dispatch to ForwardDiff (CPU-only)
@inline function gradient(f::F, x::SVector{N,T}) where {F,N,T}
    seeds = ntuple(i -> Dual{_InnerTag}(x[i], ntuple(j -> ifelse(j==i, one(T), zero(T)), Val(N))), Val(N))
    y = f(SVector(seeds))
    SVector(ntuple(j -> _ip(y, j), Val(N)))
end
@inline function jacobian(f::F, x::SVector{N,T}) where {F,N,T}
    seeds = ntuple(i -> Dual{_InnerTag}(x[i], ntuple(j -> ifelse(j==i, one(T), zero(T)), Val(N))), Val(N))
    _stack_jac(f(SVector(seeds)), Val(N))
end
@inline function _stack_jac(ydual::SVector{M}, ::Val{N}) where {M,N}
    SMatrix{M,N}(ntuple(k -> _ip(ydual[((k-1) % M) + 1], ((k-1) ÷ M) + 1), Val(M*N)))
end
@inline derivative(f::F, t::T) where {F,T} = map(yi -> _ip(yi, 1), f(Dual{_InnerTag}(t, one(T))))
@inline gradient(f, x) = ForwardDiff.gradient(f, x)
@inline jacobian(f, x) = ForwardDiff.jacobian(f, x)

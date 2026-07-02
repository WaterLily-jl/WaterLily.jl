using KernelAbstractions: get_backend, @index, @kernel
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
    @log "p/c, iter, r∞, r₂, ω\n"
end

@inline CI(a...) = CartesianIndex(a...)
"""
    CIj(j,I,k)
Replace jᵗʰ component of CartesianIndex with k
"""
CIj(j,I::CartesianIndex{d},k) where d = CI(ntuple(i -> i==j ? k : I[i], d))

"""
    δ(i,N::Int)
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

    @simd for I ∈ R
        @fastmath @inbounds a[I,i] += sum(loc(i,I))
    end

on serial execution, or

    @kernel function kern(a,i,@Const(offset),@Const(I0))
        I ∈ @index(Global,Cartesian)+I0
        @fastmath @inbounds a[I,i] += sum(loc(i,I,offset))
    end
    kern(get_backend(a),64)(a,i,offset,R[1]-oneunit(R[1]),ndrange=size(R))

when multi-threading on CPU or using CuArrays.  The macro rewrites every
`loc(...)` call in `expr` to append a captured `offset` argument so that
`loc` returns *global* coordinates in MPI-parallel runs; in serial the
captured value is `nothing` and `loc(...,nothing)` falls back to local
coordinates.  `get_backend` is used on the _first_ variable in `expr`.
"""
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args
    sym = []
    grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    symT = [gensym() for _ in 1:length(sym)] # generate a list of types for each symbol
    symWtypes = joinsymtype(rep.(sym),symT) # symbols with types: [a::A, b::B, ...]
    @gensym(kern, kern_, offset) # unique kernel names + captured offset symbol
    inject_loc_offset!(ex, offset) # rewrite loc(...) → loc(..., offset) in ex
    @static if backend == "KernelAbstractions"
        return quote
            local $offset = WaterLily._loop_offset(eltype($(sym[1])))
            @kernel function $kern_($(symWtypes...),@Const($offset),@Const(I0)) where {$(symT...)}
                $I = @index(Global,Cartesian)
                $I += I0
                @fastmath @inbounds $ex
            end
            function $kern($(symWtypes...),$offset) where {$(symT...)}
                $kern_(get_backend($(sym[1])),64)($(sym...),$offset,$R[1]-oneunit($R[1]),ndrange=size($R))
            end
            $kern($(sym...),$offset)
        end |> esc
    else # backend == "SIMD"
        return quote
            local $offset = WaterLily._loop_offset(eltype($(sym[1])))
            function $kern($(symWtypes...),$offset) where {$(symT...)}
                @simd for $I ∈ $R
                    @fastmath @inbounds $ex
                end
            end
            $kern($(sym...),$offset)
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

# Walk `ex` and append `offset` to every bare `loc(...)` call so positions
# inside a @loop body are in global coordinates.  The rewrite uses
# `GlobalRef(@__MODULE__, :loc)` so the call resolves unambiguously to
# WaterLily.loc, even if the caller's module does not `using WaterLily:loc`.
# Scope-aware: `function`, lambda (`->`), `let`, and `for` scopes that bind
# the symbol `loc` disable the rewrite within their body.  Qualified names
# like `WaterLily.loc` are untouched because their head is `:.`, not `:call`.
function inject_loc_offset!(ex, offset, shadowed::Bool=false)
    ex isa Expr || return ex
    h = ex.head
    if h === :function || h === :->
        inner = shadowed || _binds_loc(ex.args[1])
        inject_loc_offset!(ex.args[end], offset, inner)
        return ex
    elseif h === :(=) && ex.args[1] isa Expr && ex.args[1].head === :call
        # short-form fn def: f(x) = body
        inner = shadowed || _binds_loc(ex.args[1])
        inject_loc_offset!(ex.args[2], offset, inner)
        return ex
    elseif h === :let || h === :for
        bind, body = ex.args[1], ex.args[end]
        inject_loc_offset!(bind, offset, shadowed)
        inner = shadowed || _binds_loc(bind)
        inject_loc_offset!(body, offset, inner)
        return ex
    end
    if h === :call && ex.args[1] === :loc && !shadowed
        ex.args[1] = GlobalRef(@__MODULE__, :loc)
        push!(ex.args, offset)
    end
    foreach(a -> inject_loc_offset!(a, offset, shadowed), ex.args)
    return ex
end

# Does this LHS expression bind the symbol `loc`?  Used to detect scopes
# (function/lambda params, let/for bindings) that shadow WaterLily.loc.
_binds_loc(s::Symbol) = s === :loc
function _binds_loc(ex::Expr)
    h = ex.head
    if h === :tuple || h === :block || h === :call
        any(_binds_loc, ex.args)
    elseif h === :(::) || h === :(=) || h === :kw || h === :...
        _binds_loc(ex.args[1])
    else
        false
    end
end
_binds_loc(_) = false

using StaticArrays
"""
    loc(i,I) = loc(Ii)

Location in space of the cell at CartesianIndex `I` at face `i`.
Using `i=0` returns the cell center s.t. `loc = I`.

Inside a `@loop` body the macro automatically appends the MPI rank-local
offset so `loc(...)` returns *global* coordinates — user code is identical
in serial and parallel.  Outside `@loop`, `loc(...)` returns rank-local
coordinates; add `global_offset(Val(N), T)` explicitly to get global ones.
"""
@inline loc(i,I::CartesianIndex{N},T::Type=Float32) where N = SVector{N,T}(I.I .- T(1.5) .- δ(i,I).I ./T(2))
@inline loc(Ii::CartesianIndex,T::Type=Float32) = loc(last(Ii),Base.front(Ii),T)
# SVector offset overloads — used by the @loop auto-injection
@inline loc(i,I::CartesianIndex{N},offset::SVector{N,T}) where {N,T} = loc(i,I,T) + offset
@inline loc(Ii::CartesianIndex,offset::SVector) = loc(last(Ii),Base.front(Ii),eltype(offset)) + offset
@inline loc(i,I::CartesianIndex{N},T::Type,offset::SVector{N}) where N = loc(i,I,T) + offset
@inline loc(Ii::CartesianIndex,T::Type,offset::SVector) = loc(last(Ii),Base.front(Ii),T) + offset
# Nothing sentinel: serial @loop passes `nothing` — no-ops to the plain loc
@inline loc(i,I::CartesianIndex,::Nothing) = loc(i,I)
@inline loc(Ii::CartesianIndex,::Nothing) = loc(Ii)
@inline loc(i,I::CartesianIndex,T::Type,::Nothing) = loc(i,I,T)
@inline loc(Ii::CartesianIndex,T::Type,::Nothing) = loc(Ii,T)
Base.last(I::CartesianIndex) = last(I.I)
Base.front(I::CartesianIndex) = CI(Base.front(I.I))
"""
    slice(dims,i,j,low=1)

Return `CartesianIndices` range slicing through an array of size `dims` in
dimension `j` at index `i` (or range `i`). `low` optionally sets the lower
extent of the range in the other dimensions.
"""
function slice(dims::NTuple{N},i,j,low=1) where N
    CartesianIndices(ntuple( k-> k==j ? (i:i) : (low:dims[k]), N))
end
function slice(dims::NTuple{N},i::AbstractUnitRange,j,low=1) where N
    CartesianIndices(ntuple( k-> k==j ? i : (low:dims[k]), N))
end

"""
    BC!(a, uBC, saveexit=false, perdir=(), t=0)

Apply domain boundary conditions to the ghost cells of a _vector_ field.
A Dirichlet condition is applied to the _normal_ component; zero Neumann to tangential.
Periodic directions are handled by `velocity_comm!` (called at the end),
separating domain BCs from communication BCs.  Under MPI the `phys_left`/
`phys_right` gates skip physical writes at rank-internal faces (the halo
exchange supplies neighbor data there instead).
"""
BC!(a,U,saveexit=false,perdir=(),t=0) = BC!(a,(i,x,t)->U[i],saveexit,perdir,t)
BC!(a,uBC::Function,saveexit=false,perdir=(),t=0) = _BC!(a, uBC, saveexit, perdir, t, par_mode[])
function _BC!(a, uBC::Function, saveexit, perdir, t, ::AbstractParMode)
    N,n = size_u(a)
    for j ∈ 1:n
        j in perdir && continue  # periodic handled by velocity_comm!
        L, R = phys_left(j), phys_right(j)
        for i ∈ 1:n
            if i==j # Normal direction, Dirichlet
                L && @loop a[I,i] = uBC(i,loc(i,I),t) over I ∈ slice(N,1:2,j)
                R && (!saveexit || i>1) && (@loop a[I,i] = uBC(i,loc(i,I),t) over I ∈ slice(N,N[j],j))
            else    # Tangential directions, Neumann: mirror
                L && @loop a[I,i] = uBC(i,loc(i,I),t)+a[I+δ(j,I),i]-uBC(i,loc(i,I+δ(j,I)),t) over I ∈ slice(N,1,j)
                R && @loop a[I,i] = uBC(i,loc(i,I),t)+a[I-δ(j,I),i]-uBC(i,loc(i,I-δ(j,I)),t) over I ∈ slice(N,N[j],j)
            end
        end
    end
    velocity_comm!(a, perdir)
end

"""
    exitBC!(u,u⁰,Δt)

Apply a 1D convection scheme to fill the ghost cell on the exit of the domain.
"""
exitBC!(u,u⁰,Δt) = _exitBC!(u,u⁰,Δt,par_mode[])
function _exitBC!(u,u⁰,Δt,::AbstractParMode)
    N,_ = size_u(u)
    inflowL = slice(N.-1,2,1,2)              # inflow face (left wall face)
    exitR   = slice(N.-1,N[1],1,2)           # exit slice excluding ghosts (right wall face)
    L, R = phys_left(1), phys_right(1)       # ranks owning the inflow / exit faces
    glen = global_allreduce(R ? length(exitR) : 0)                          # global exit-face length
    U = global_allreduce(L ? sum(@view u[inflowL,1]) : zero(eltype(u)))/glen # inflow mass flux
    R && @loop u[I,1] = u⁰[I,1]-U*Δt*(u⁰[I,1]-u⁰[I-δ(1,I),1]) over I ∈ exitR
    ∮u = global_allreduce(R ? sum(@view u[exitR,1]) : zero(eltype(u)))/glen - U  # mass flux imbalance
    R && @loop u[I,1] -= ∮u over I ∈ exitR   # correct flux
    velocity_halo!(u)                        # no-op serial; MPI exchange parallel
end
"""
    perBC!(a,perdir)

Apply periodic conditions to the ghost cells of a _scalar_ field.
"""
perBC!(a,::Tuple{}) = nothing
perBC!(a, perdir, N = size(a)) = for j ∈ perdir
    @loop a[I] = a[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
    @loop a[I] = a[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
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
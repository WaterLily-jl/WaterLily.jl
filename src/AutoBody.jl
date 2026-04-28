"""
    AutoBody(sdf,map=(x,t)->x) <: AbstractBody

  - `sdf(x::AbstractVector,t::Real)::Real`: signed distance function
  - `map(x::AbstractVector,t::Real)::AbstractVector`: coordinate mapping function

Implicitly define a geometry by its `sdf` and optional coordinate `map`. Note: the `map`
is composed automatically i.e. `sdf(body::AutoBody,x,t) = body.sdf(body.map(x,t),t)`.
"""
struct AutoBody{F1<:Function,F2<:Function} <: AbstractBody
    sdf::F1
    map::F2
end
AutoBody(sdf, map=(x,t)->x) = AutoBody(sdf, map)

"""
    d = sdf(body::AutoBody,x,t) = body.sdf(body.map(x,t),t)
"""
@inline sdf(body::AutoBody,x,t=0;kwargs...) = body.sdf(body.map(x,t),t)

using ForwardDiff
using ForwardDiff: Dual, partials, Tag

# Inner-derivative tag for measure's gradient/jacobian/derivative on `body.sdf`
# and `body.map`. Compile-time singleton: `≺` is overloaded so it always ranks
# strictly newer than any outer `ForwardDiff.Tag`. This lets nested AD work on
# the GPU — the stock ForwardDiff path goes through `extract_jacobian`/`valtype`,
# which calls `tagcount` (a side-effecting generated function) at runtime; on
# GPU the codegen can instantiate the inner tag's count before the host
# instantiates the outer's, inverting the precedence and triggering
# `DualMismatchError` inside the kernel.
struct _InnerTag end
@inline ForwardDiff.:≺(::Type{<:Tag}, ::Type{_InnerTag}) = true
@inline ForwardDiff.:≺(::Type{_InnerTag}, ::Type{<:Tag}) = false
@inline ForwardDiff.:≺(::Type{_InnerTag}, ::Type{_InnerTag}) = false

# Tag-aware partial extractor: returns the i-th partial only if `y` carries an
# `_InnerTag` dual; otherwise the function did not depend on the seeded input
# and the inner derivative is exactly zero. Without this guard a non-_InnerTag
# `y` (which could still be an outer-tag `Dual` pulled in by closure capture)
# would silently return the outer partial.
@inline _ip(y::Dual{_InnerTag}, i::Int) = partials(y, i)
@inline _ip(y, ::Int) = zero(y)

# Tag-stable, GPU-safe gradient/jacobian/derivative on SVector inputs. They
# extract `partials` directly so neither `extract_jacobian` nor `valtype` is hit.
@inline function _gradient(f::F, x::SVector{N,T}) where {F,N,T}
    seeds = ntuple(i -> Dual{_InnerTag}(x[i], ntuple(j -> ifelse(j==i, one(T), zero(T)), Val(N))), Val(N))
    y = f(SVector(seeds))
    SVector(ntuple(j -> _ip(y, j), Val(N)))
end
@inline function _jacobian(f::F, x::SVector{N,T}) where {F,N,T}
    seeds = ntuple(i -> Dual{_InnerTag}(x[i], ntuple(j -> ifelse(j==i, one(T), zero(T)), Val(N))), Val(N))
    _stack_jac(f(SVector(seeds)), Val(N))
end
@inline function _stack_jac(ydual::SVector{M}, ::Val{N}) where {M,N}
    SMatrix{M,N}(ntuple(k -> _ip(ydual[((k-1) % M) + 1], ((k-1) ÷ M) + 1), Val(M*N)))
end
@inline _derivative(f::F, t::T) where {F,T} = map(yi -> _ip(yi, 1), f(Dual{_InnerTag}(t, one(T))))

# Dispatch wrappers: the SVector path is GPU-safe, but `measure` may also be
# called with plain `AbstractVector` inputs (e.g. user-facing tests). For non-
# SVector inputs we fall back to stock ForwardDiff — this slow path doesn't
# enter GPU kernels so the nested-FD GPU bug doesn't apply.
@inline _grad(f, x::SVector) = _gradient(f, x)
@inline _grad(f, x) = ForwardDiff.gradient(f, x)
@inline _jac(f, x::SVector) = _jacobian(f, x)
@inline _jac(f, x) = ForwardDiff.jacobian(f, x)

"""
    d,n,V = measure(body::AutoBody,x,t;fastd²=Inf)

Determine the implicit geometric properties from the `sdf` and `map`.
The gradient of `d=sdf(map(x,t))` is used to improve `d` for pseudo-sdfs.
The velocity is determined _solely_ from the optional `map` function.
Skips the `n,V` calculation when `d²>fastd²`.
"""
function measure(body::AutoBody,x,t;fastd²=Inf)
    d = sdf(body,x,t)
    d^2>fastd² && return (d,zero(x),zero(x))
    n = _grad(ξ->body.sdf(ξ,t), body.map(x,t)) # body-frame only
    any(isnan, n) && return (d,zero(x),zero(x)) # handle non-diff'able points
    J = _jac(x->body.map(x,t), x)               # for mapping n,V to x-frame
    n = J'n; m = √sum(abs2,n); d /= m; n /= m   # chain rule then normalise
    return (d, n, -J\_derivative(t->body.map(x,t), t))
end

using LinearAlgebra: tr
"""
    curvature(A::AbstractMatrix)

Return `H,K` the mean and Gaussian curvature from `A=hessian(sdf)`.
`K=tr(minor(A))` in 3D and `K=0` in 2D.
"""
function curvature(A::AbstractMatrix)
    H,K = 0.5*tr(A),0
    if size(A)==(3,3)
        K = A[1,1]*A[2,2]+A[1,1]*A[3,3]+A[2,2]*A[3,3]-A[1,2]^2-A[1,3]^2-A[2,3]^2
    end
    H,K
end

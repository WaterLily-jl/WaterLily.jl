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
    d = sdf(body::AutoBody,x,t) = body.sdf(body.map(x+offset,t),t)

The local coordinate `x` is shifted by `global_offset` so that the user's
SDF and map functions always receive global coordinates.  In serial the
offset is zero; in MPI it is the rank-local origin.
"""
@inline sdf(body::AutoBody,x::SVector{N},t=0;kwargs...) where N =
    body.sdf(body.map(x .+ global_offset(Val(N)),t),t)

using ForwardDiff
"""
    d,n,V = measure(body::AutoBody,x,t;fastd²=Inf)

Determine the implicit geometric properties from the `sdf` and `map`.
The local coordinate `x` is shifted by `global_offset` so that the user's
SDF and map functions always receive global coordinates.
The gradient of `d=sdf(map(xg,t))` is used to improve `d` for pseudo-sdfs.
The velocity is determined _solely_ from the optional `map` function.
Skips the `n,V` calculation when `d²>fastd²`.
"""
function measure(body::AutoBody,x::SVector{N},t;fastd²=Inf) where N
    # shift to global coordinates (zero in serial, rank offset in MPI)
    xg = x .+ global_offset(Val(N))
    # eval d=f(xg,t), and n̂ = ∇f
    d = body.sdf(body.map(xg,t),t)
    d^2>fastd² && return (d,zero(x),zero(x)) # skip n,V
    n = ForwardDiff.gradient(x->body.sdf(body.map(x,t),t), xg)
    any(isnan.(n)) && return (d,zero(x),zero(x))

    # correct general implicit fnc f(x₀)=0 to be a pseudo-sdf
    #   f(x) = f(x₀)+d|∇f|+O(d²) ∴  d ≈ f(x)/|∇f|
    m = √sum(abs2,n); d /= m; n /= m

    # The velocity depends on the material change of ξ=m(x,t):
    #   Dm/Dt=0 → ṁ + (dm/dx)ẋ = 0 ∴  ẋ =-(dm/dx)\ṁ
    J = ForwardDiff.jacobian(x->body.map(x,t), xg)
    dot = ForwardDiff.derivative(t->body.map(xg,t), t)
    return (d,n,-J\dot)
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

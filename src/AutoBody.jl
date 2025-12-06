"""
    AutoBody(sdf,map=(x,t)->x; compose=true) <: AbstractBody

  - `sdf(x::AbstractVector,t::Real)::Real`: signed distance function
  - `map(x::AbstractVector,t::Real)::AbstractVector`: coordinate mapping function
  - `compose::Bool=true`: Flag for composing `sdf=sdf∘map`

Implicitly define a geometry by its `sdf` and optional coordinate `map`. Note: the `map`
is composed automatically if `compose=true`, i.e. `sdf(x,t) = sdf(map(x,t),t)`.
Both parameters remain independent otherwise. It can be particularly heplful to set
`compose=false` when adding mulitple bodies together to create a more complex one.
"""
struct AutoBody{F1<:Function,F2<:Function} <: AbstractBody
    sdf::F1
    map::F2
    function AutoBody(sdf, map=(x,t)->x; compose=true)
        comp(x,t) = compose ? sdf(map(x,t),t) : sdf(x,t)
        new{typeof(comp),typeof(map)}(comp, map)
    end
end

"""
    d = sdf(body::AutoBody,x,t) = body.sdf(x,t)
"""
sdf(body::AutoBody,x,t=0;kwargs...) = body.sdf(x,t)

using ForwardDiff
"""
    d,n,V = measure(body::AutoBody,x,t;fastd²=Inf)

Determine the implicit geometric properties from the `sdf` and `map`.
The gradient of `d=sdf(map(x,t))` is used to improve `d` for pseudo-sdfs.
The velocity is determined _solely_ from the optional `map` function.
Skips the `n,V` calculation when `d²>fastd²`.
"""
function measure(body::AutoBody,x,t;fastd²=Inf)
    # eval d=f(x,t), and n̂ = ∇f
    d = body.sdf(x,t)
    d^2>fastd² && return (d,zero(x),zero(x)) # skip n,V
    n = ForwardDiff.gradient(x->body.sdf(x,t), x)
    any(isnan.(n)) && return (d,zero(x),zero(x))

    # correct general implicit fnc f(x₀)=0 to be a pseudo-sdf
    #   f(x) = f(x₀)+d|∇f|+O(d²) ∴  d ≈ f(x)/|∇f|
    m = √sum(abs2,n); d /= m; n /= m

    # get velocity from map
    return (d,n,velocity(body.map, x, t))
end

# The velocity depends on the material change of ξ=m(x,t):
#   Dm/Dt=0 → ṁ + (dm/dx)ẋ = 0 ∴  ẋ =-(dm/dx)\ṁ
velocity(map::Function, x, t) = -ForwardDiff.jacobian(x->map(x,t), x)\ForwardDiff.derivative(t->map(x,t), t)


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

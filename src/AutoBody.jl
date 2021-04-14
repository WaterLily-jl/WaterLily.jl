"""
    AutoBody(sdf,map=(x,t)->x) <: AbstractBody

    - sdf(x::AbstractVector,t::Real)::Real: signed distance function
    - map(x::AbstractVector,t::Real)::AbstractVector: coordinate mapping function

Define a geometry by its `sdf` and optional coordinate `map`. All other
properties are determined using Automatic Differentiation. Note: the `map`
is composed automatically if provided, ie `sdf(x,t) = sdf(map(x,t),t)`.
"""
struct AutoBody{F1<:Function,F2<:Function} <: AbstractBody
    sdf::F1
    map::F2
    function AutoBody(sdf,map=(x,t)->x)
        comp(x,t) = sdf(map(x,t),t)
        new{typeof(comp),typeof(map)}(comp, map)
    end
end

using ForwardDiff,DiffResults
using LinearAlgebra: norm2
"""
    measure(body::AutoBody,x::Vector,t::Real)

ForwardDiff is used to determine the geometric properties from the `sdf`.
Note: The velocity is determined _soley_ from the optional `map` function.
"""
function measure(body::AutoBody,x::AbstractVector,t::Real)
    # V = dot(map), ignoring any other time dependancy in sdf.
    V = -ForwardDiff.derivative(t->body.map(x,t), t)

    # Use DiffResults to get Hessian/gradient/value in one shot
    result = DiffResults.HessianResult(x)
    result = ForwardDiff.hessian!(result, x -> body.sdf(x,t), x)
    d = Float64(DiffResults.value(result))
    n̂ = DiffResults.gradient(result)
    H,K = curvature(DiffResults.hessian(result))

    # |∇d|=1 for a true distance function. If sdf or map violates this condition,
    # scaling by |∇d| gives an approximate correction.
    m = norm2(n̂)
    d/m,n̂./m,V,H/m,K/m^2
end

using LinearAlgebra: tr
"""
    curvature(A::AbstractMatrix)

Return `H,K` the mean and Gaussian curvature from `A=hessian(sdf)`.
K=tr(minor(A)) in 3D and K=0 in 2D.
"""
function curvature(A::AbstractMatrix)
    H,K = 0.5*tr(A),0
    if size(A)==(3,3)
        K = A[1,1]*A[2,2]+A[1,1]*A[3,3]+A[2,2]*A[3,3]-A[1,2]^2-A[1,3]^2-A[2,3]^2
    end
    H,K
end

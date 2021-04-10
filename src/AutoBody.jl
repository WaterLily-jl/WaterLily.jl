"""
    AutoBody(sdf,map=(x,t)->x) <: AbstractBody

    - sdf(x::Vector,t::Real)::Real: signed distance function
    - map(x::Vector,t::Real)::Vector: coordinate mapping function

Define a geometry by its `sdf` and optional coordinate `map`. All other
properties are determined using Automatic Differentiation. Note: the `map`
is composed automatically if provided, ie `sdf(x,t) = sdf(map(x,t),t)`.
"""
struct AutoBody <: AbstractBody
    sdf::Function
    map::Function
    function AutoBody(sdf,map=(x,t)->x)
        new((x,t)->sdf(map(x,t),t),map)
    end
end

using ForwardDiff,DiffResults
using LinearAlgebra: norm2
"""
    measure(body::AutoBody,x::Vector,t::Real)

ForwardDiff is used to determine the geometric properties from the `sdf`.
Note: The velocity is determined _soley_ from the optional `map` function.
"""
function measure(body::AutoBody,x::Vector,t::Real)
    # V = dot(map), ignoring any other time dependancy in sdf.
    V = -ForwardDiff.derivative(t->body.map(x,t), t)

    # Use DiffResults to get Hessian/gradient/value in one shot
    result = DiffResults.HessianResult(x)
    result = ForwardDiff.hessian!(result, x -> body.sdf(x,t), x)
    d = Float64(DiffResults.value(result))
    n̂ = DiffResults.gradient(result)
    κ = curvature(DiffResults.hessian(result))

    # |∇d|=1 for a true distance function. If sdf or map violates this condition,
    # scaling by |∇d| gives an approximate correction.
    m = norm2(n̂)
    d/m,n̂./m,κ./[m,m^2],V
end

using LinearAlgebra: tr
"""
    curvature(A::Matrix)

Return `κ = [H,K]` the mean and Gaussian curvature from `A=hessian(sdf)`.
K=tr(minor(A)) in 3D and K=0 in 2D.
"""
function curvature(A::Matrix)
    H,K = 0.5*tr(A),0
    if size(A)==(3,3)
        K = A[1,1]*A[2,2]+A[1,1]*A[3,3]+A[2,2]*A[3,3]-A[1,2]^2-A[1,3]^2-A[2,3]^2
    end
    [H,K]
end

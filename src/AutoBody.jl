"""
    AutoBody

Defines the geometry using only a time variable signed distance function `sdf`.
Other properties are determined using Automatic Differentiation.
"""
struct AutoBody <: AbstractBody
    sdf::Function
end

using ForwardDiff,DiffResults
"""
    measure(body::AutoBody,x::Vector,t::Real)

ForwardDiff is used to determine the geometry properties from the `sdf`.
Note: V set to 0 for now.
"""
function measure(body::AutoBody,x::Vector,t::Real)
    result = DiffResults.HessianResult(x)
    result = ForwardDiff.hessian!(result, x -> body.sdf(x,t), x)
    d = DiffResults.value(result)
    n̂ = DiffResults.gradient(result)
    κ = curvature(DiffResults.hessian(result))
    V = zeros(length(x)) #-ForwardDiff.derivative(t->body.sdf(x,t), t) .* n̂
    d,n̂,κ,V
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

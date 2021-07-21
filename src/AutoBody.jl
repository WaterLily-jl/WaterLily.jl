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

using StaticArrays
"""
    measure!(flow::Flow, body::AutoBody; t=0, ϵ=1)

Uses `body.sdf` and `body.map` to fill the arrays:

    `flow.μ₀`, Zeroth kernel moment
    `flow.μ₁`, First kernel moment scaled by the body normal
    `flow.V`,  Body velocity
    `flow.σᵥ`,  Body velocity divergence scaled by `μ₀-1`

at time `t` using an immersion kernel of size `ϵ`.
See Maertens & Weymouth, https://doi.org/10.1016/j.cma.2014.09.007
"""
function measure!(a::Flow{N},body::AutoBody;t=0,ϵ=1) where N
    a.V .= 0; a.μ₀ .= 1; a.μ₁ .= 0
    for I ∈ inside(a.p)
        x = SVector(I.I...)       # location at cell center
        d = body.sdf(x,t)
        if abs(d)<ϵ+1             # only measure near interface
            for i ∈ 1:N
                xᵢ = x .-0.5.*δ(i,N).I    # location at face
                dᵢ,n,V,H,K = measure(body,xᵢ,t)
                a.V[I,i] = V[i]
                a.μ₀[I,i] = μ₀(dᵢ,ϵ)
                a.μ₁[I,i,:] = μ₁(dᵢ,ϵ).*n
            end
        elseif d<0
            a.μ₀[I,:] .= 0
        end
        a.σᵥ[I] = μ₀(d,ϵ)-1
    end
    @inside a.σᵥ[I] = a.σᵥ[I]*div(I,a.V)
    BC!(a.μ₀,zeros(N))
end

using ForwardDiff,DiffResults
"""
    measure(body::AutoBody,x,t)

ForwardDiff is used to determine the geometric properties from the `sdf`.
Note: The velocity is determined _soley_ from the optional `map` function.
"""
function measure(body::AutoBody,x::AbstractVector,t::Real)
    # V = dot(map), ignoring any other time dependancy in sdf.
    V = -ForwardDiff.derivative(t->body.map(x,t), t)

    # Use DiffResults to get Hessian/gradient/value in one shot
    result = DiffResults.HessianResult(x)
    result = ForwardDiff.hessian!(result, x -> body.sdf(x,t), x)
    d = DiffResults.value(result)
    n̂ = DiffResults.gradient(result)
    H,K = curvature(DiffResults.hessian(result))

    # |∇d|=1 for a true distance function. If sdf or map violates this condition,
    # scaling by |∇d| gives an approximate correction.
    m = √sum(abs2,n̂)
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

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

"""
    measure!(flow::Flow, body::AutoBody; t=0, ϵ=1)

Uses `body.sdf` and `body.map` to fill the arrays:

    `flow.μ₀`, Zeroth kernel moment
    `flow.μ₁`, First kernel moment scaled by the body normal
    `flow.V`,  Body velocity
    `flow.σᵥ`,  Body velocity divergence scaled by `μ₀-1`

at time `t` using an immersion kernel of size `ϵ`.
See [Maertens & Weymouth](https://eprints.soton.ac.uk/369635/)
"""
function measure!(a::Flow{N},body::AutoBody;t=0,ϵ=1) where N
    a.V .= 0; a.μ₀ .= 1; a.μ₁ .= 0; a.σᵥ .= 0
    sdf = a.σ
    apply_sdf!(x->body.sdf(x,t), sdf)   # distance to cell center
    for I ∈ inside(a.p)
        if abs(sdf[I])<ϵ+1   # near interface
            a.σᵥ[I] = μ₀(sdf[I],ϵ)-1
            # measure properties and fill arrays at face (i,I)
            for i ∈ 1:N
                dᵢ,n,V = measure(body,loc(i,I),t)
                m = √sum(abs2,n); dᵢ /= m; n /= m
                a.V[I,i] = V[i]
                a.μ₀[I,i] = μ₀(dᵢ,ϵ)
                a.μ₁[I,i,:] = μ₁(dᵢ,ϵ).*n
            end
        elseif sdf[I]<0      # completely inside body
            a.μ₀[I,:] .= 0
            a.σᵥ[I] = -1
        end
    end
    @inside a.σᵥ[I] = a.σᵥ[I]*div(I,a.V) # scaled divergence
    BC!(a.μ₀,zeros(SVector{N}))
end

function apply_sdf!(f,a,margin=2,stride=1)
    # strided index and signed distance function
    @inline J(I) = stride*I+oneunit(I)
    @inline sdf(I) = f(WaterLily.loc(0,J(I)) .- (stride-1)/2)

    # if the strided array is indivisible, fill it using the sdf 
    dims = (size(a) .-2 ) .÷ stride
    if sum(mod.(dims,2)) != 0
        @loop a[J(I)] = sdf(I) over I ∈ CartesianIndices(dims)
    
    # if not, fill an array with twice the stride first
    else    
        apply_sdf!(f,a,margin,2stride)

    # and only improve the values within a margin of sdf=0
        tol = stride*(√length(dims)+margin)
        @loop (d = @inbounds a[J(I)+stride*CI(mod.(I.I,2))]; 
               a[J(I)] = abs(d)<tol ? sdf(I) : d) over I ∈ CartesianIndices(dims)
    end
end

using ForwardDiff
"""
    measure(body::AutoBody,x,t)

ForwardDiff is used to determine the geometric properties from the `sdf`.
Note: The velocity is determined _soley_ from the optional `map` function.
"""
measure(body,x,t) = (body.sdf(x,t),                               # d
                    ForwardDiff.gradient(x->body.sdf(x,t), x),    # n=∇d
                    -ForwardDiff.derivative(t->body.map(x,t), t)) # V = -dot(X)

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

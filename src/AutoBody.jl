"""
    AutoBody(sdf,map=(x,t)->x; compose=true) <: AbstractBody

    - sdf(x::AbstractVector,t::Real)::Real: signed distance function
    - map(x::AbstractVector,t::Real)::AbstractVector: coordinate mapping function
    - compose::Bool: if true, automatically compose the `map`, ie `sdf(x,t) = sdf(map(x,t),t)`
                        else, `sdf` and `map` remain independent

Define a geometry by its `sdf` and optional coordinate `map`. All other
properties are determined using Automatic Differentiation. Note: the `map`
is composed automatically if compose is set to `true`, ie `sdf(x,t) = sdf(map(x,t),t)`. 
Both parameters remain independent otherwise. It can be particularly heplful to set it as 
false when adding mulitple bodies together to create a more complexe one.
"""
struct AutoBody{F1<:Function,F2<:Function} <: AbstractBody
    sdf::F1
    map::F2
    function AutoBody(sdf, map=(x,t)->x; compose=true)
        comp(x,t) = compose ? sdf(map(x,t),t) : sdf(x,t)
        new{typeof(comp),typeof(map)}(comp, map)
    end
end

function Base.:+(a::AutoBody, b::AutoBody)
    map(x,t) = ifelse(a.sdf(x,t)<b.sdf(x,t),a.map(x,t),b.map(x,t))
    sdf(x,t) = min(a.sdf(x,t),b.sdf(x,t))
    AutoBody(sdf,map,compose=false)
end
function Base.:∩(a::AutoBody, b::AutoBody)
    map(x,t) = ifelse(a.sdf(x,t)>b.sdf(x,t),a.map(x,t),b.map(x,t))
    sdf(x,t) = max(a.sdf(x,t),b.sdf(x,t))
    AutoBody(sdf,map,compose=false)
end
Base.:∪(x::AutoBody, y::AutoBody) = x+y
Base.:-(x::AutoBody) = AutoBody((d,t)->-x.sdf(d,t),x.map,compose=false)
Base.:-(x::AutoBody, y::AutoBody) = x ∩ -y

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
    fast_sdf!(a.σ,body,t) # distance to cell center
    @fastmath @inline function fill!(μ₀,μ₁,V,σᵥ,d,I)
        σᵥ[I] = WaterLily.μ₀(d[I],ϵ)-1 # cell-center array
        if abs(d[I])<1+ϵ
            for i ∈ 1:N
                dᵢ,nᵢ,Vᵢ = measure(body,WaterLily.loc(i,I),t)
                V[I,i] = Vᵢ[i]
                μ₀[I,i] = WaterLily.μ₀(dᵢ,ϵ)
                for j ∈ 1:N
                    μ₁[I,i,j] = WaterLily.μ₁(dᵢ,ϵ)*nᵢ[j]
                end
            end
        elseif d[I]<0
            for i ∈ 1:N
                μ₀[I,i] = 0.
            end
        end
    end
    @loop fill!(a.μ₀,a.μ₁,a.V,a.σᵥ,a.σ,I) over I ∈ inside(a.p)
    @inside a.σᵥ[I] = a.σᵥ[I]*div(I,a.V)              # scaled divergence
    correct_div!(a.σᵥ)
    BC!(a.μ₀,zeros(SVector{N}))                       # fill BCs
end

fast_sdf!(a::AbstractArray,body::AutoBody,t) = fast_sdf!(x->body.sdf(x,t),a)
function fast_sdf!(f::Function,a::AbstractArray,margin=2,stride=1)
    # strided index and signed distance function
    @inline J(I) = stride*I+oneunit(I)
    @inline sdf(I) = f(loc(0,J(I)) .- (stride-1)/2)
    @inline mod2(I) = CI(mod.(I.I,2))

    # if the strided array is indivisible, fill it using the sdf 
    dims = (size(a) .-2 ) .÷ stride
    if sum(mod.(dims,2)) != 0
        @loop a[J(I)] = sdf(I) over I ∈ CartesianIndices(dims)
    
    # if not, fill an array with twice the stride first
    else    
        fast_sdf!(f,a,margin,2stride)
        @loop a[J(I)] = a[J(I)+stride*mod2(I)] over I ∈ CartesianIndices(dims)

    # and only improve the values within a margin of sdf=0
        tol = stride*(√length(dims)+margin)
        @loop a[J(I)] = abs(a[J(I)])<tol ? sdf(I) : a[J(I)] over I ∈ CartesianIndices(dims)
    end
end

using ForwardDiff
"""
    measure(body::AutoBody,x,t)

ForwardDiff is used to determine the geometric properties from the `sdf`.
Note: The velocity is determined _soley_ from the optional `map` function.
"""
function measure(body,x,t)
    # eval d=f(x,t), and n̂ = ∇f
    d = body.sdf(x,t)
    n = ForwardDiff.gradient(x->body.sdf(x,t), x)

    # correct general implicit fnc f(x₀)=0 to be a psuedo-sdf 
    #   f(x) = f(x₀)+d|∇f|+O(d²) ∴  d ≈ f(x)/|∇f|
    m = √sum(abs2,n); d /= m; n /= m

    # The velocity depends on the material change of ξ=m(x,t):
    #   Dm/Dt=0 → ṁ + (dm/dx)ẋ = 0 ∴  ẋ =-(dm/dx)\ṁ
    J = ForwardDiff.jacobian(x->body.map(x,t), x)
    dot = ForwardDiff.derivative(t->body.map(x,t), t)
    return (d,n,-J\dot)
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

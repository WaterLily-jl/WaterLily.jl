using StaticArrays
"""
    AbstractBody

Immersed body Abstract Type. Any `AbstractBody` subtype must implement

    d,n,V = measure(body::AbstractBody, x, t=0, fastd²=Inf)

where `d` is the signed distance from `x` to the body at time `t`,
and `n` & `V` are the normal and velocity vectors implied at `x`.
A fast-approximate method can return `≈d,zero(x),zero(x)` if `d^2>fastd²`.
"""
abstract type AbstractBody end
"""
    measure!(flow::Flow, body::AbstractBody; t=0, ϵ=1)

Queries the body geometry to fill the arrays:

- `flow.μ₀`, Zeroth kernel moment
- `flow.μ₁`, First kernel moment scaled by the body normal
- `flow.V`,  Body velocity

at time `t` using an immersion kernel of size `ϵ`.

See Maertens & Weymouth, doi:[10.1016/j.cma.2014.09.007](https://doi.org/10.1016/j.cma.2014.09.007).
"""
function measure!(a::Flow{N,T},body::AbstractBody;t=zero(T),ϵ=1) where {N,T}
    a.V .= zero(T); a.μ₀ .= one(T); a.μ₁ .= zero(T); d²=(2+ϵ)^2
    @fastmath @inline function fill!(μ₀,μ₁,V,d,I)
        d[I] = sdf(body,loc(0,I,T),t,fastd²=d²)
        if d[I]^2<d²
            for i ∈ 1:N
                dᵢ,nᵢ,Vᵢ = measure(body,loc(i,I,T),t,fastd²=d²)
                V[I,i] = Vᵢ[i]
                μ₀[I,i] = WaterLily.μ₀(dᵢ,ϵ)
                for j ∈ 1:N
                    μ₁[I,i,j] = WaterLily.μ₁(dᵢ,ϵ)*nᵢ[j]
                end
            end
        elseif d[I]<zero(T)
            for i ∈ 1:N
                μ₀[I,i] = zero(T)
            end
        end
    end
    @loop fill!(a.μ₀,a.μ₁,a.V,a.σ,I) over I ∈ inside(a.p)
    BC!(a.μ₀,zeros(SVector{N,T}),false,a.perdir) # BC on μ₀, don't fill normal component yet
    BC!(a.V ,zeros(SVector{N,T}),a.exitBC,a.perdir)
end

# Convolution kernel and its moments
@fastmath kern(d) = 0.5+0.5cos(π*d)
@fastmath kern₀(d) = 0.5+0.5d+0.5sin(π*d)/π
@fastmath kern₁(d) = 0.25*(1-d^2)-0.5*(d*sin(π*d)+(1+cos(π*d))/π)/π

μ₀(d,ϵ) = kern₀(clamp(d/ϵ,-1,1))
μ₁(d,ϵ) = ϵ*kern₁(clamp(d/ϵ,-1,1))

"""
    d = sdf(a::AbstractBody,x,t=0;fastd²=0)

Measure only the distance. Defaults to fastd²=0 for quick evaluation.
"""
sdf(body::AbstractBody,x,t=0;fastd²=0) = measure(body,x,t;fastd²)[1]

"""
    measure_sdf!(a::AbstractArray, body::AbstractBody, t=0; fastd²=0)

Uses `sdf(body,x,t)` to fill `a`. Defaults to fastd²=0 for quick evaluation.
"""
measure_sdf!(a::AbstractArray{T},body::AbstractBody,t=zero(T);fastd²=zero(T)) where T = @inside a[I] = sdf(body,loc(0,I,T),t;fastd²)

"""
    NoBody

Use for a simulation without a body.
"""
struct NoBody <: AbstractBody end
measure(::NoBody,x::AbstractVector,args...;kwargs...)=(Inf,zero(x),zero(x))
function measure!(::Flow,::NoBody;kwargs...) end # skip measure! entirely

"""
    SetBody

Body defined as a lazy set operation on two `AbstractBody`s.
The operations are only evaluated when `measure`d.
"""
struct SetBody{O<:Function,Ta<:AbstractBody,Tb<:AbstractBody} <: AbstractBody
    op::O
    a::Ta
    b::Tb
end

# Lazy constructors
Base.:∪(a::AbstractBody, b::AbstractBody) = SetBody(min,a,b)
Base.:+(a::AbstractBody, b::AbstractBody) = a ∪ b
Base.:∩(a::AbstractBody, b::AbstractBody) = SetBody(max,a,b)
Base.:-(a::AbstractBody) = SetBody(-,a,NoBody())
Base.:-(a::AbstractBody, b::AbstractBody) = a ∩ (-b)

# Measurements
function measure(body::SetBody,x,t;fastd²=Inf)
    body.op(measure(body.a,x,t;fastd²),measure(body.b,x,t;fastd²)) # can't mapreduce within GPU kernel
end
measure(body::SetBody{typeof(-)},x,t;fastd²=Inf) = ((d,n,V) = measure(body.a,x,t;fastd²); (-d,-n,V))
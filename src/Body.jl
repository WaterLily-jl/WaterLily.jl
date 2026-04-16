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

at time `t` using an immersion kernel of size `ϵ`. The velocity is only filled within a narrow band
of size `2+ϵ` around the body. This function also fills `flow.σ` with the signed distance function.

See Maertens & Weymouth, doi:[10.1016/j.cma.2014.09.007](https://doi.org/10.1016/j.cma.2014.09.007).
"""
function measure!(a::Flow{N,T},body::AbstractBody;t=zero(T),ϵ=1) where {N,T}
    a.V .= zero(T); a.μ₀ .= one(T); a.μ₁ .= zero(T); d²=T(2+ϵ)^2
    measure_sdf!(a.σ, body, t; fastd²=d²) # measure separately to allow specialization
    @fastmath @inline function fill!(μ₀,μ₁,V,d,I)
        if d[I]^2<d²
            for i ∈ 1:N
                dᵢ,nᵢ,Vᵢ = measure(body,loc(i,I,T),t,fastd²=d²)
                dᵢ = abs(dᵢ) ≤ 0.5 ? dᵢ : copysign(dᵢ,d[I]) # enforce sign consistency
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
    BC!(a.μ₀,zeros(SVector{N,T}),false,a.perdir)
    BC!(a.V ,zeros(SVector{N,T}),a.exitBC,a.perdir)
    velocity_halo!(reshape(a.μ₁, size(a.σ)..., :)) # halo on μ₁ tensor (no-op in serial)
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
function measure!(a::Flow{N,T},::NoBody;kwargs...) where {N,T}
    a.μ₀ .= one(T)
    BC!(a.μ₀,zeros(SVector{N,T}),false,a.perdir)
end

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

"""
    OffsetBody(body, offset) <: AbstractBody

Wraps any `AbstractBody` so that rank-local coordinates are shifted by
`offset` to global coordinates before `measure`/`sdf` evaluation.
Created automatically by `_apply_offset`; users should not construct directly.
"""
struct OffsetBody{B<:AbstractBody,O} <: AbstractBody
    body::B
    offset::O
end
measure(b::OffsetBody,x,t;kwargs...) = measure(b.body, x .+ b.offset, t; kwargs...)
sdf(b::OffsetBody,x,t=0;kwargs...) = sdf(b.body, x .+ b.offset, t; kwargs...)

"""
    _apply_offset(body::AbstractBody, offset)

Wrap the body in an `OffsetBody` so that rank-local coordinates are shifted
by `offset` before evaluation.  Works for any `AbstractBody` subtype.
"""
_apply_offset(body::AbstractBody, offset) = OffsetBody(body, offset)
_apply_offset(body::NoBody, offset) = body

# Lazy constructors
Base.:∪(a::AbstractBody, b::AbstractBody) = SetBody(min,a,b)
Base.:+(a::AbstractBody, b::AbstractBody) = a ∪ b
Base.:∩(a::AbstractBody, b::AbstractBody) = SetBody(max,a,b)
Base.:-(a::AbstractBody) = SetBody(-,a,NoBody())
Base.:-(a::AbstractBody, b::AbstractBody) = a ∩ (-b)

# Measurements
function measure(body::SetBody,x::AbstractVector{T},t;fastd²=T(Inf)) where T 
    body.op(measure(body.a,x,t;fastd²),measure(body.b,x,t;fastd²)) # can't mapreduce within GPU kernel
end
measure(body::SetBody{typeof(-)},x::AbstractVector{T},t;fastd²=T(Inf)) where T = ((d,n,V) = measure(body.a,x,t;fastd²); (-d,-n,V))
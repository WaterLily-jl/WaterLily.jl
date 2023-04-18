using StaticArrays
"""
    AbstractBody

Immersed body Abstract Type. Any `AbstractBody` subtype must implement

    `measure!(flow::Flow, body::AbstractBody; t=0, ϵ=1)`

which queries the body geometry to fill the arrays:

    `flow.μ₀`, Zeroth kernel moment
    `flow.μ₁`, First kernel moment scaled by the body normal
    `flow.V`,  Body velocity
    `flow.σᵥ`, Body velocity divergence scaled by `μ₀-1`

at time `t` using an immersion kernel of size `ϵ`.
See Maertens & Weymouth, https://doi.org/10.1016/j.cma.2014.09.007
"""
abstract type AbstractBody end

# Convolution kernel and its moments
@fastmath kern(d) = 0.5+0.5cos(π*d)
@fastmath kern₀(d) = 0.5+0.5d+0.5sin(π*d)/π
@fastmath kern₁(d) = 0.25*(1-d^2)-0.5*(d*sin(π*d)+(1+cos(π*d))/π)/π

μ₀(d,ϵ) = kern₀(clamp(d/ϵ,-1,1))
μ₁(d,ϵ) = ϵ*kern₁(clamp(d/ϵ,-1,1))

"""
    NoBody

Use for a simulation without a body
"""
struct NoBody <: AbstractBody end
function measure!(a::Flow,body::NoBody;t=0,ϵ=1) end

function correct_div!(σ)
    s = sum(σ)/length(inside(σ))
    abs(s) <= 2eps(eltype(s)) && return
    @inside σ[I] = σ[I]-s
end
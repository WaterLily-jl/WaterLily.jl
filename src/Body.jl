"""
    AbstractBody

Abstract body geometry data structure. This solver will call

    `measure(body::AbstractBody,x::Vector,t::Real)`

to query the body geometry for these properties at location `x` and time `t`:

    `d :: Real`, Signed distance to surface
    `n̂ :: Vector`, Outward facing unit normal
    `κ :: Vector`, Mean and Gaussian curvature
    `V :: Vector`, Body velocity

"""
abstract type AbstractBody end

# Convolution kernel and its moments
kern(d) = 0.5+0.5cos(π*d)
kern₀(d) = 0.5+0.5d+0.5sin(π*d)/π
kern₁(d) = 0.25*(d^2-1)+0.5*(d*sin(π*d)+(1+cos(π*d))/π)/π

clamp1(x) = clamp(x,-1,1)
μ₀(d;ϵ=1) = d/ϵ |> clamp1 |> kern₀

"""
    measure(a::Flow,body::AbstractBody;t=0,ϵ=1)

Measure the `body` properties on `flow` using a kernel
size `ϵ`. Weymouth & Yue, JCP, 2011
"""
function measure!(a::Flow,body::AbstractBody;t=0,ϵ=1)
    N = size(a.μ₀)
    for b ∈ 1:N[end]
        @simd for I ∈ CR(N[1:end-1])
            x = collect(Float16, I.I) # location at cell center
            x[b] -= 0.5               # location at face
            d,n̂,κ,V = measure(body,x,t)
            @inbounds a.μ₀[I,b] = μ₀(d;ϵ)
            @inbounds a.V[I,b] = V[b]
        end
    end
    BC!(a.μ₀,zeros(m))
end

"""
    NoBody

Use for a simulation without a body
"""
struct NoBody <: AbstractBody end
function measure!(a::Flow,body::NoBody;t=0,ϵ=1) end

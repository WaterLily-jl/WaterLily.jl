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
kern₁(d) = 0.25*(1-d^2)-0.5*(d*sin(π*d)+(1+cos(π*d))/π)/π

clamp1(x) = clamp(x,-1,1)
μ₀(d;ϵ=1) = kern₀(clamp1(d/ϵ))
μ₁(d;ϵ=1) = ϵ*kern₁(clamp1(d/ϵ))

"""
    measure(a::Flow,body::AbstractBody;t=0,ϵ=1)

Measure the `body` properties on `flow` using a kernel
size `ϵ`. Weymouth & Yue, JCP, 2011
"""
function measure!(a::Flow{N},body::AbstractBody;t=0,ϵ=1) where N
    a.V .= 0; a.μ₀ .= 0; a.μ₁ .= 0
    for I ∈ inside(a.p)
        x = collect(Float16, I.I) # location at cell center
        d = body.sdf(x .-1/3,t)
        if abs(d)<ϵ+0.5           # only measure near interface
            for i ∈ 1:N
                xᵢ=x; xᵢ[i] -= 0.5  # location at face
                dᵢ,n,κ,V = measure(body,xᵢ,t)
                a.V[I,i] = V[i]
                a.μ₀[I,i] = μ₀(dᵢ;ϵ)
                a.μ₁[I,i,:] = μ₁(dᵢ;ϵ).*n
            end
        elseif d>0
            a.μ₀[I,:] .= 1
        end
    end
    BC!(a.μ₀,zeros(N))
end
"""
    NoBody

Use for a simulation without a body
"""
struct NoBody <: AbstractBody end
function measure!(a::Flow,body::NoBody;t=0,ϵ=1) end

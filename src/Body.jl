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

"""
    apply(f, N...)

Apply a vector function f(i,x) to the faces of a uniform staggard grid.
"""
function apply(f,N...)
    # TODO be more clever with the type
    c = Array{Float64}(undef,N...)
    for b ∈ 1:N[end]
        @simd for I ∈ CR(N[1:end-1])
            x = collect(Float16, I.I) # location at cell center
            x[b] -= 0.5               # location at face
            @inbounds c[I,b] = f(b,x) # apply function to location
        end
    end
    return c
end

"""
    BDIM_coef(f, N...)

Compute the boundary data immersion method coefficients `c`
given a signed distance function `f`. Weymouth & Yue, JCP, 2011
"""
BDIM_coef(f,N...) = apply((i,x)->f(x),N...) .|> clamp1 .|> kern₀

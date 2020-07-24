# Convolution kernel and its moments
kern(d) = 0.5+0.5cos(π*d)
kern₀(d) = 0.5+0.5d+0.5sin(π*d)/π
kern₁(d) = 0.25*(d^2-1)+0.5*(d*sin(π*d)+(1+cos(π*d))/π)/π
"""
    BDIM_coef

Compute the boundary data immersion method coefficients `c`
given a signed distance function `f`. Weymouth & Yue, JCP, 2011
"""
function BDIM_coef(f,N...)
    c = Array{Float64}(undef,N...)
    BDIM_coef!(f,c)
    return c
end
@fastmath function BDIM_coef!(f,c)
    N = size(c)
    for b ∈ 1:N[end]
        @simd for I ∈ inside_u(N)
            x = collect(Float16, I.I) # location at cell center
            x[b] -= 0.5      # location at face
            d = f(x)         # signed distance function
            @inbounds c[I,b] = kern₀(clamp(d,-1,1))
        end
    end
    BC!(c,zeros(N[end]))
end

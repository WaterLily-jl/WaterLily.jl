"""
    PoissonSys{N,M}

Composite type for conservative variable coefficient Poisson equations:

    ∮ds β ∂x/∂n = σ

The resulting linear system is

    Ax = [L+D+L']x = b

where A is symmetric, block-tridiagonal and extremely sparse. Implemented on a
structured grid of dimension M, then L has dimension N=M+1 and size(L,N)=M.
Moreover, D[I]=-∑ᵢ(L[I,i]+L'[I,i]). This means matrix storage, multiplication,
ect can be easily implemented and optimized without external libraries.

To help iteratively solve the system above, the PoissonSys structure holds
helper arrays for inv(D), the error ϵ=x̂=x, and residual r=b-Ax. An iterative
solution method then estimates the error ϵ=̃A⁻¹r and increments x+=ϵ, r-=Aϵ.
"""
struct PoissonSys{N,M}
    L :: Array{Float64,N} # Lower diagonal coefficients
    D :: Array{Float64,M} # Diagonal coefficients
    iD :: Array{Float64,M} # 1/Diagonal
    x :: Array{Float64,M} # approximate solution
    ϵ :: Array{Float64,M} # increment/error
    r :: Array{Float64,M} # residual
    function PoissonSys(L::Array{Float64,n}) where n
        N = size(L); M = N[1:end-1]; m = length(M)
        @assert N[end] == m
        x,ϵ,r,D,iD = AU(M),AU(M),AU(M),AU(M),AU(M)
        for I ∈ inside(M)
            D[I] = -sum(i->L[I,i]+L[I+δ(i,m),i],1:m)
            iD[I] = abs2(D[I])<1e-8 ? 0. : inv(D[I])
        end
        new{n,m}(L,D,iD,x,ϵ,r)
    end
end

@fastmath @inline function multLU(I::CartesianIndex{d},L,x) where d
    s = 0
    for i ∈ 1:d
        @inbounds s += x[I-δ(i,I)]*L[I,i]+x[I+δ(i,I)]*L[I+δ(i,I),i]
    end
    return s
end
@fastmath @inline @inbounds mult(I,L,D,x) = multLU(I,L,x)+x[I]*D[I]
mult!(p::PoissonSys) = @simd for I ∈ inside(p.r)
    @inbounds p.r[I] = mult(I,p.L,p.D,p.x)
end

@fastmath function L₂(a::Array{Float64})
    s = 0.
    @simd for I ∈ inside(a)
        @inbounds s += abs2(a[I])
    end
    return s
end

@fastmath residual!(p::PoissonSys,b::Array{Float64}) = @simd for I ∈ inside(p.r)
    @inbounds p.r[I] = b[I]-mult(I,p.L,p.D,p.x)
end

@fastmath increment!(p::PoissonSys) = @simd for I ∈ inside(p.x)
    @inbounds p.x[I] += p.ϵ[I]
    @inbounds p.r[I] -= mult(I,p.L,p.D,p.ϵ)
end
"""
SOR!(p::PoissonSys;ω=1.5)

Successive Over Relaxation preconditioner. The routine uses backsubstitution
to compute ϵ=̃A⁻¹r, where ̃A=[D/ω+L], and then increments x,r.
"""
@fastmath function SOR!(p::PoissonSys{n,m}; ω=1.5) where {n,m}
    for I ∈ inside(p.r) # order matters here
        @inbounds σ = p.r[I]
        for i ∈ 1:m
            @inbounds σ -= p.L[I,i]*p.ϵ[I-δ(i,I)]
        end
        @inbounds p.ϵ[I] = ω*σ*p.iD[I]
    end
    increment!(p)
end

function solve!(x::Array{Float64,m},p::PoissonSys{n,m},b::Array{Float64,m};log=false,tol=1e-4) where {n,m}
    p.x .= x
    residual!(p,b); r₂ = L₂(p.r)
    log && (res = [r₂])
    while r₂>tol
        SOR!(p,ω=1.8); r₂ = L₂(p.r)
        log && push!(res,r₂)
    end
    x .= p.x
    return log ? (x,res) : x
end

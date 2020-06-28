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
abstract type Poisson{N,M} end
struct PoissonSys{N,M} <: Poisson{N,M}
    L :: Array{Float64,N} # Lower diagonal coefficients
    D :: Array{Float64,M} # Diagonal coefficients
    iD :: Array{Float64,M} # 1/Diagonal
    x :: Array{Float64,M} # approximate solution
    ϵ :: Array{Float64,M} # increment/error
    r :: Array{Float64,M} # residual
    function PoissonSys(L::Array{Float64,n}) where n
        N = size(L); M = N[1:end-1]; m = length(M)
        @assert N[end] == m
        x,ϵ,r,D,iD = zeros(M),zeros(M),zeros(M),zeros(M),zeros(M)
        for I ∈ inside(M)
            D[I] = -sum(i->L[I,i]+L[I+δ(i,m),i],1:m)
            iD[I] = abs2(D[I])<1e-8 ? 0. : inv(D[I])
        end
        new{n,m}(L,D,iD,x,ϵ,r)
    end
end

@fastmath @inline multL(a,I,L,x) = @inbounds x[I-δ(a,I)]*L[I,a]
@fastmath @inline multU(a,I,L,x) = @inbounds x[I+δ(a,I)]*L[I+δ(a,I),a]
@fastmath @inline function mult(I::CartesianIndex{d},L,D,x) where d
    σ = @inbounds x[I]*D[I]
    @simd for a ∈ 1:d
        σ += multL(a,I,L,x)
        σ += multU(a,I,L,x)
    end
    return σ
end
function mult(p::PoissonSys{n,m},x::Array{Float64,m}) where {n,m}
    @assert size(p.x)==size(x)
    b = zeros(size(p.x))
    @simd for I ∈ inside(b)
        @inbounds b[I] = mult(I,p.L,p.D,x)
    end
    return b
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
        for a ∈ 1:m
            σ -= multL(a,I,p.L,p.ϵ)
        end
        @inbounds p.ϵ[I] = ω*σ*p.iD[I]
    end
    increment!(p)
end
"""
GS!(p::PoissonSys;it=0)

Gauss-Sidel smoother. When it=0, the function serves as a Jacobi preconditioner.
"""
@fastmath function GS!(p::PoissonSys{n,m};it=0) where {n,m}
    @simd for I ∈ inside(p.r)
        @inbounds p.ϵ[I] = p.r[I]*p.iD[I]
    end
    for i ∈ 1:it; for I ∈ inside(p.r, reverse = i%2==0) # order matters here
        @inbounds σ = p.r[I]
        for a ∈ 1:m
            σ -= multL(a,I,p.L,p.ϵ)
            σ -= multU(a,I,p.L,p.ϵ)
        end
        @inbounds p.ϵ[I] = σ*p.iD[I]
    end;end
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
    return log ? res : nothing
end

"""
    Poisson{N,M}

Composite type for conservative variable coefficient Poisson equations:

    ∮ds β ∂x/∂n = σ

The resulting linear system is

    Ax = [L+D+L']x = b

where A is symmetric, block-tridiagonal and extremely sparse. Implemented on a
structured grid of dimension N, then L has dimension M=N+1 and size(L,M)=N.
Moreover, D[I]=-∑ᵢ(L[I,i]+L'[I,i]). This means matrix storage, multiplication,
ect can be easily implemented and optimized without external libraries.

To help iteratively solve the system above, the Poisson structure holds
helper arrays for inv(D), the error ϵ, and residual r=b-Ax. An iterative
solution method then estimates the error ϵ=̃A⁻¹r and increments x+=ϵ, r-=Aϵ.
"""
abstract type AbstractPoisson{N,M} end
struct Poisson{N,M} <: AbstractPoisson{N,M}
    L :: Array{Float64,M} # Lower diagonal coefficients
    D :: Array{Float64,N} # Diagonal coefficients
    iD :: Array{Float64,N} # 1/Diagonal
    x :: Array{Float64,N} # approximate solution
    ϵ :: Array{Float64,N} # increment/error
    r :: Array{Float64,N} # residual
    function Poisson(L::Array{Float64,m}) where m
        M = size(L); N = M[1:end-1]; n = m-1
        @assert M[end] == n
        x,ϵ,r,D,iD = zeros(N),zeros(N),zeros(N),zeros(N),zeros(N)
        set_diag!(D,iD,L)
        new{n,m}(L,D,iD,x,ϵ,r)
    end
end
function set_diag!(D,iD,L)
    @inbounds @simd for I ∈ inside(D)
        D[I] = -sum(@inbounds(L[I,i]+L[I+δ(i,n),i]) for i ∈ 1:n)
        iD[I] = abs2(D[I])<1e-8 ? 0. : inv(D[I])
    end
end
set_diag!(p::Poisson) = set_diag!(p.D,p.iD,p.L)
update!(p::Poisson,L::Array) = (p.L .= L; set_diag!(p))

@fastmath @inline multL(I::CartesianIndex{d},L,x) where {d} =
    sum(@inbounds(x[I-δ(a,I)]*L[I,a]) for a ∈ 1:d)
@fastmath @inline multU(I::CartesianIndex{d},L,x) where {d} =
    sum(@inbounds(x[I+δ(a,I)]*L[I+δ(a,I),a]) for a ∈ 1:d)
@fastmath @inline mult(I,L,D,x) = @inbounds(x[I]*D[I])+multL(I,L,x)+multU(I,L,x)
function mult(p::Poisson{n},x::Array{Float64,n}) where n
    @assert size(p.x)==size(x)
    b = zeros(size(p.x))
    @inside b[I] = mult(I,p.L,p.D,x)
    return b
end

@fastmath residual!(p::Poisson,b::Array{Float64}) =
    @inside p.r[I] = b[I]-mult(I,p.L,p.D,p.x)

@fastmath increment!(p::Poisson) = @inbounds @simd for I ∈ inside(p.x)
    p.x[I] += p.ϵ[I]
    p.r[I] -= mult(I,p.L,p.D,p.ϵ)
end
"""
SOR!(p::Poisson;ω=1.5)

Successive Over Relaxation preconditioner. The routine uses backsubstitution
to compute ϵ=̃A⁻¹r, where ̃A=[D/ω+L], and then increments x,r.
"""
@fastmath function SOR!(p::Poisson; ω=1.5)
    @inbounds for I ∈ inside(p.r) # order matters here
        σ = p.r[I]-multL(I,p.L,p.ϵ)
        p.ϵ[I] = ω*σ*p.iD[I]
    end
    increment!(p)
end
"""
GS!(p::Poisson;it=0)

Gauss-Sidel smoother. When it=0, the function serves as a Jacobi preconditioner.
"""
@fastmath function GS!(p::Poisson;it=0)
    @inside p.ϵ[I] = p.r[I]*p.iD[I]
    for i ∈ 1:it; @inbounds for I ∈ inside(p.r, reverse = i%2==0) # order matters here
        σ = p.r[I]-multL(I,p.L,p.ϵ)-multU(I,p.L,p.ϵ)
        p.ϵ[I] = σ*p.iD[I]
    end;end
    increment!(p)
end

function solve!(x::Array{Float64,n},p::Poisson{n},b::Array{Float64,n};log=false,tol=1e-4,itmx=1e3) where n
    p.x .= x
    residual!(p,b); r₂ = L₂(p.r)
    log && (res = [r₂])
    nᵖ=0
    while r₂>tol && nᵖ<itmx
        SOR!(p,ω=1.8); r₂ = L₂(p.r)
        log && push!(res,r₂)
        nᵖ+=1
    end
    x .= p.x
    return log ? res : nᵖ
end

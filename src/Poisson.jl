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
abstract type AbstractPoisson{N,M,T} end
struct Poisson{N,M,T} <: AbstractPoisson{N,M,T}
    L :: Array{T,M} # Lower diagonal coefficients
    D :: Array{T,N} # Diagonal coefficients
    iD :: Array{T,N} # 1/Diagonal
    x :: Array{T,N} # approximate solution
    ϵ :: Array{T,N} # increment/error
    r :: Array{T,N} # residual
    n :: Vector{Int16}    # pressure solver iterations
    function Poisson(L::AbstractArray{T}) where T
        N,n = size_u(L)
        x,ϵ,r,D,iD = zeros(T,N),zeros(T,N),zeros(T,N),zeros(T,N),zeros(T,N)
        set_diag!(D,iD,L)
        new{n,n+1,T}(L,D,iD,x,ϵ,r,[])
    end
end

@fastmath @inline diag(I::CartesianIndex{d},L) where {d} =
    -sum(@inbounds(L[I,i]+L[I+δ(i,I),i]) for i ∈ 1:d)
function set_diag!(D,iD,L)
    @inside D[I] = diag(I,L)
    @inside iD[I] = abs2(D[I])<1e-8 ? 0. : inv(D[I])
end
set_diag!(p::Poisson) = set_diag!(p.D,p.iD,p.L)
update!(p::Poisson,L) = (p.L .= L; set_diag!(p))

@fastmath @inline multL(I::CartesianIndex{d},L,x) where {d} =
    sum(@inbounds(x[I-δ(a,I)]*L[I,a]) for a ∈ 1:d)
@fastmath @inline multU(I::CartesianIndex{d},L,x) where {d} =
    sum(@inbounds(x[I+δ(a,I)]*L[I+δ(a,I),a]) for a ∈ 1:d)
@fastmath @inline mult(I,L,D,x) = @inbounds(x[I]*D[I])+multL(I,L,x)+multU(I,L,x)

"""
    mult(A::AbstractPoisson,x)

Efficient function for Poisson matrix-vector multiplication. Allocates and returns
`b = Ax` with `b=0` in the ghost cells.
"""
function mult(p::Poisson,x)
    @assert size(p.x)==size(x)
    b = zeros(size(p.x))
    @inside b[I] = mult(I,p.L,p.D,x)
    return b
end

@fastmath residual!(p::Poisson,b) =
    @inside p.r[I] = b[I]-mult(I,p.L,p.D,p.x)

@fastmath function increment!(p::Poisson)
    @inside p.x[I] = p.x[I]+p.ϵ[I]
    @inside p.r[I] = p.r[I]-mult(I,p.L,p.D,p.ϵ)
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
    for i ∈ 1:it
        @inside p.ϵ[I] = p.iD[I]*(p.r[I]-multL(I,p.L,p.ϵ)-multU(I,p.L,p.ϵ))
    end
    increment!(p)
end

"""
    solver!(x,A::AbstractPoisson,b;log,tol,itmx)

Approximate iterative solver for the Poisson matrix equation `Ax=b`.

    `x`: Initial-solution vector mutated by `solver!`
    `A`: Poisson matrix
    `b`: Right-Hand-Side vector
    `log`: If `true`, this function returns a vector holding the `L₂`-norm of the residual at each iteration.
    `tol`: Convergence tolerance on the `L₂`-norm residual.
    'itmx': Maximum number of iterations
"""
function solver!(x,p::Poisson,b;log=false,tol=1e-4,itmx=1e3)
    @assert size(p.x)==size(x)
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
    _ENABLE_PUSH && push!(p.n,nᵖ)
    log && return res
end

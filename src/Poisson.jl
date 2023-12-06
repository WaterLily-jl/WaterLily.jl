"""
    Poisson{N,M}

Composite type for conservative variable coefficient Poisson equations:

    ∮ds β ∂x/∂n = σ

The resulting linear system is

    Ax = [L+D+L']x = z

where A is symmetric, block-tridiagonal and extremely sparse. Moreover, 
`D[I]=-∑ᵢ(L[I,i]+L'[I,i])`. This means matrix storage, multiplication,
ect can be easily implemented and optimized without external libraries.

To help iteratively solve the system above, the Poisson structure holds
helper arrays for `inv(D)`, the error `ϵ`, and residual `r=z-Ax`. An iterative
solution method then estimates the error `ϵ=̃A⁻¹r` and increments `x+=ϵ`, `r-=Aϵ`.
"""
abstract type AbstractPoisson{T,S,V} end
struct Poisson{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    L :: V # Lower diagonal coefficients
    D :: S # Diagonal coefficients
    iD :: S # 1/Diagonal
    x :: S # approximate solution
    ϵ :: S # increment/error
    r :: S # residual
    z :: S # source
    n :: Vector{Int16} # pressure solver iterations
    perdir :: NTuple # direction of periodic boundary condition
    function Poisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};perdir=(0,)) where T
        @assert axes(x) == axes(z) && axes(x) == Base.front(axes(L)) && last(axes(L)) == eachindex(axes(x))
        r = similar(x); fill!(r,0)
        ϵ,D,iD = copy(r),copy(r),copy(r)
        set_diag!(D,iD,L)
        new{T,typeof(x),typeof(L)}(L,D,iD,x,ϵ,r,z,[],perdir)
    end
end

function set_diag!(D,iD,L)
    @inside D[I] = diag(I,L)
    @inside iD[I] = abs2(D[I])<1e-8 ? 0. : inv(D[I])
end
update!(p::Poisson) = set_diag!(p.D,p.iD,p.L)

@fastmath @inline function diag(I::CartesianIndex{d},L) where {d}
    s = zero(eltype(L))
    for i in 1:d
        s -= @inbounds(L[I,i]+L[I+δ(i,I),i])
    end
    return s
end

"""
    mult!(p::Poisson,x)

Efficient function for Poisson matrix-vector multiplication. 
Fills `p.z = p.A x` with 0 in the ghost cells.
"""
function mult!(p::Poisson,x)
    @assert axes(p.z)==axes(x)
    fill!(p.z,0)
    @inside p.z[I] = mult(I,p.L,p.D,x)
    return p.z
end
@fastmath @inline function mult(I::CartesianIndex{d},L,D,x) where {d}
    s = @inbounds(x[I]*D[I])
    for i in 1:d
        s += @inbounds(x[I-δ(i,I)]*L[I,i]+x[I+δ(i,I)]*L[I+δ(i,I),i])
    end
    return s
end

residual!(p::Poisson) = @inside p.r[I] = p.z[I]-mult(I,p.L,p.D,p.x)

increment!(p::Poisson) = @loop (p.r[I] = p.r[I]-mult(I,p.L,p.D,p.ϵ);
                                p.x[I] = p.x[I]+p.ϵ[I]) over I ∈ inside(p.x)
"""
    Jacobi!(p::Poisson; it=1)

Jacobi smoother run `it` times. 
Note: This runs for general backends, but is _very_ slow to converge.
"""
@fastmath Jacobi!(p;it=1) = for _ ∈ 1:it
    @inside p.ϵ[I] = p.r[I]*p.iD[I]
    BC!(p.ϵ;perdir=p.perdir)
    increment!(p)
end

using LinearAlgebra: ⋅
"""
    pcg!(p::Poisson; it=6)

Conjugate-Gradient smoother with Jacobi preditioning. Runs at most `it` iterations, 
but will exit early if the Gram-Schmidt update parameter `|α| < 1%` or `|r D⁻¹ r| < 1e-8`.
Note: This runs for general backends and is the default smoother.
"""
function pcg!(p::Poisson;it=6)
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    insideI = inside(x) # [insideI]
    rho = r ⋅ z
    abs(rho)<1e-12 && return
    for i in 1:it
        BC!(ϵ;perdir=p.perdir)
        @inside z[I] = mult(I,p.L,p.D,ϵ)
        alpha = rho/(z[insideI]⋅ϵ[insideI])
        @loop (x[I] += alpha*ϵ[I];
               r[I] -= alpha*z[I]) over I ∈ inside(x)
        (i==it || abs(alpha)<1e-2) && return
        @inside z[I] = r[I]*p.iD[I]
        rho2 = r⋅z
        abs(rho2)<1e-8 && return
        beta = rho2/rho
        @inside ϵ[I] = beta*ϵ[I]+z[I]
        rho = rho2        
    end
end
smooth!(p) = pcg!(p)
# smooth!(p) = get_backend(p.r)==CPU() ? SOR!(p,it=3) : Jacobi!(p,it=20)

L₂(p::Poisson) = p.r ⋅ p.r # special method since outside(p.r)≡0

"""
    solver!(A::Poisson;log,tol,itmx)

Approximate iterative solver for the Poisson matrix equation `Ax=b`.

  - `A`: Poisson matrix with working arrays.
  - `A.x`: Solution vector. Can start with an initial guess.
  - `A.z`: Right-Hand-Side vector. Will be overwritten!
  - `A.n[end]`: stores the number of iterations performed.
  - `log`: If `true`, this function returns a vector holding the `L₂`-norm of the residual at each iteration.
  - `tol`: Convergence tolerance on the `L₂`-norm residual.
  - `itmx`: Maximum number of iterations.
"""
function solver!(p::Poisson;log=false,tol=1e-4,itmx=1e3)
    BC!(p.x;perdir=p.perdir)
    residual!(p); r₂ = L₂(p)
    log && (res = [r₂])
    nᵖ=0
    while r₂>tol && nᵖ<itmx
        smooth!(p); r₂ = L₂(p)
        log && push!(res,r₂)
        nᵖ+=1
    end
    BC!(p.x;perdir=p.perdir)
    push!(p.n,nᵖ)
    log && return res
end

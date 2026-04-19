abstract type AbstractPoisson{T,S,V} end

"""
    Poisson{T, S, V}

Composite type for conservative variable coefficient Poisson equations:

    ∮ds β ∂x/∂n = σ

The resulting linear system is

    Ax = [L+D+L']x = z

where A is symmetric, block-tridiagonal and extremely sparse. Moreover,
`D[I]=-∑ᵢ(L[I,i]+L'[I,i])`. This means matrix storage, multiplication,
etc. can be easily implemented and optimized without external libraries.

The conductivity `L` is initialized from `flow.μ₀` and modified by
`wallBC_L!` (zero at physical walls for Neumann BCs). Ghost-cell
synchronization uses `comm!` (= `perBC!` + `scalar_halo!`) rather than
bare `perBC!`, so that MPI rank-internal boundaries are handled correctly.

To help iteratively solve the system, the structure holds helper arrays
for `inv(D)`, the error `ϵ`, and residual `r=z-Ax`. An iterative solution
method estimates the error `ϵ≈A⁻¹r` and increments `x+=ϵ`, `r-=Aϵ`.
The solver ends with `pin_pressure!` + `comm!` to remove the null-space
mode and synchronize halos.
"""
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
    inslen :: Int # global number of inside cells (precomputed to avoid Allreduce)
    function Poisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};perdir=()) where T
        @assert axes(x) == axes(z) && axes(x) == Base.front(axes(L)) && last(axes(L)) == eachindex(axes(x))
        r = similar(x); fill!(r,0)
        ϵ,D,iD = copy(r),copy(r),copy(r)
        set_diag!(D,iD,L)
        new{T,typeof(x),typeof(L)}(L,D,iD,x,ϵ,r,z,[],perdir,global_length(inside(x)))
    end
end

using ForwardDiff: Dual,Tag
Base.eps(::Type{D}) where D<:Dual{Tag{G,T}} where {G,T} = eps(T)
function set_diag!(D,iD,L)
    @inside D[I] = diag(I,L)
    # Precision-independent threshold: in F64, 2eps(T)≈4e-16 is too tight and
    # leaves tiny-D interface cells with huge iD that drift under multigrid.
    @inside iD[I] = abs2(D[I])<2eps(Float32) ? 0. : inv(D[I])
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
Fills `p.z = Ax` with 0 in the ghost cells, where `A` is the Poisson matrix implied by `L` and `D`.
"""
function mult!(p::Poisson,x)
    @assert axes(p.z)==axes(x)
    comm!(x,p.perdir)
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

"""
    residual!(p::Poisson)

Computes the residual `r = z-Ax` and corrects it such that
`r = 0` if `iD==0` which ensures local satisfiability
    and
`sum(r) = 0` which ensures global satisfiability.

The global correction is done by adjusting all points uniformly,
minimizing the local effect. Other approaches are possible.

Note: These corrections mean `x` is not strictly solving `Ax=z`, but
without the corrections, no solution exists.
"""
function residual!(p::Poisson)
    comm!(p.x,p.perdir)
    @inside p.r[I] = ifelse(p.iD[I]==0,0,p.z[I]-mult(I,p.L,p.D,p.x))
    s = global_sum(p.r)/p.inslen
    abs(s) <= 2eps(eltype(s)) && return
    @inside p.r[I] = p.r[I]-s
end

function increment!(p::Poisson{T};ω=1) where {T}
    comm!(p.ϵ,p.perdir)
    @loop (p.r[I] = p.r[I]-ω*mult(I,p.L,p.D,p.ϵ);
           p.x[I] = p.x[I]+ω*p.ϵ[I]) over I ∈ inside(p.x)
end
"""
    Jacobi!(p::Poisson; it=1)

Jacobi smoother. Runs `it` iterations with relaxation parameter `ω` scaling the deferred corrections in `increment!`.
Note: This runs for general backends but converges _very_ slowly.
"""
@fastmath Jacobi!(p;it=1,ω=1) = for _ ∈ 1:it
    @inside p.ϵ[I] = p.r[I]*p.iD[I]
    increment!(p;ω)
end

@fastmath @inline function gauss(I::CartesianIndex{d},r,L,iD,x) where {d}
    s = @inbounds(r[I])
    for i in 1:d
        s -= @inbounds(x[I-δ(i,I)]*L[I,i] + x[I+δ(i,I)]*L[I+δ(i,I),i])
    end
    return s*@inbounds(iD[I])
end

@inline function gauss_rb(x,r,L,iD,k₀,Iv::CartesianIndex{d}) where {d}
    k = 2*Iv.I[end] - 1 - (sum(Base.front(Iv.I)) + k₀) % 2 # double the k-index and shift for red-black indexing
    I = CartesianIndex(ntuple( i-> i==d ? k : Iv.I[i], d))
    x[I] = gauss(I,r,L,iD,x)
end

@inline function half_rangek(x::AbstractArray{T,N}) where{T,N}
    return CartesianIndices(ntuple( i-> i==N ? (2:size(x,i)÷2) : (2:size(x,i)-1), N))
end

"""
    GaussSeidelRB!(p::Poisson;it=4, ω=1)

Red-black Gauss-Seidel smoother. Runs `it` iterations; a complete red-black cycle requires `it` to be even.
`ω` under-/over-relaxes the solution through scaling the deferred corrections in `increment!`.
Note: This performs best on GPU configurations and is the default smoother.
"""
function GaussSeidelRB!(p::Poisson{T};it=4, ω=1) where {T}
    @inside p.ϵ[I] = p.r[I]*p.iD[I]  # initialize ϵ
    comm!(p.ϵ,p.perdir)
    for i ∈ 1:it
        @loop gauss_rb(p.ϵ,p.r,p.L,p.iD,i,I) over I ∈ half_rangek(p.ϵ)
    end
    increment!(p;ω) # increment solution and residual
end

"""
    pcg!(p::Poisson; it=6)

Conjugate-Gradient smoother with Jacobi preconditioning. Runs at most `it` iterations,
but will exit early if the Gram-Schmidt update parameter `|α| < 1%` or `|r D⁻¹ r| < 1e-8`.
Note: This runs for general backends.
"""
function pcg!(p::Poisson{T};it=6,kwargs...) where T
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    rho = global_dot(r,z)
    abs(rho)<10eps(T) && return
    for i in 1:it
        comm!(ϵ,p.perdir)
        @inside z[I] = mult(I,p.L,p.D,ϵ)
        alpha = rho/global_perdot(z,ϵ,p.perdir)
        (abs(alpha)<1e-2 || abs(alpha)>1e2) && return # alpha should be O(1)
        @loop (x[I] += alpha*ϵ[I];
               r[I] -= alpha*z[I]) over I ∈ inside(x)
        i==it && return
        @inside z[I] = r[I]*p.iD[I]
        rho2 = global_dot(r,z)
        abs(rho2)<10eps(T) && return
        beta = rho2/rho
        @inside ϵ[I] = beta*ϵ[I]+z[I]
        rho = rho2
    end
end

L₂(p::Poisson) = global_dot(p.r, p.r) # special method since outside(p.r)≡0
L∞(p::Poisson) = maximum(abs,p.r)

"""
    solver!(A::Poisson; tol=1e-4, itmx=1e3)

Iterative solver for the Poisson matrix equation `Ax=b` using
preconditioned conjugate gradients (`pcg!`).

  - `A.x`: Solution vector (can start with an initial guess).
  - `A.z`: Right-hand-side vector (overwritten).
  - `A.n[end]`: Number of iterations performed.
  - `tol`: Convergence tolerance on the `L₂`-norm residual.
  - `itmx`: Maximum number of iterations.

Ends with `pin_pressure!` (remove null-space mean) and `comm!`
(halo sync) so the solution is ready for use in `project!`.
"""
function solver!(p::Poisson;tol=1e-4,itmx=1e3)
    residual!(p); r₂ = L₂(p)
    nᵖ=0; @log ", $nᵖ, $(L∞(p)), $r₂\n"
    while nᵖ<itmx
        pcg!(p); r₂ = L₂(p); nᵖ+=1
        @log ", $nᵖ, $(L∞(p)), $r₂\n"
        r₂<tol && break
    end
    pin_pressure!(p); comm!(p.x,p.perdir)
    push!(p.n,nᵖ)
end

"""
    pin_pressure!(p::Poisson)

Remove the null-space (constant) mode by subtracting the mean pressure
over fluid cells, and zero body cells (`iD==0`).  Body cells are dead
to the Poisson solve and their values are physically meaningless, so
forcing them to zero keeps serial and parallel runs bit-identical
inside the body (multigrid prolongation otherwise leaks coarse-level
solutions into fine body cells via `increment!`).
"""
function pin_pressure!(p::Poisson)
    fill!(p.z, 0)                                   # clear ghost junk (σ is shared scratch)
    @inside p.z[I] = p.x[I] * (p.iD[I] != 0)        # fluid-only mask of p
    s = global_sum(p.z) / global_allreduce(count(!=(0), p.iD))
    @inside p.x[I] = (p.x[I] - s) * (p.iD[I] != 0)  # shift fluid, zero body
end

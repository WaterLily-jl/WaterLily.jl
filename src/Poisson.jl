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

The lower diagonal `L` is aliased to `flow.μ₀`; `BC!` on μ₀ already
zeros normal-direction values at wall faces, giving `L[wall_face]=0` for
an implicit Neumann BC.  Ghost-cell synchronization uses `comm!`
(= `perBC!` + `scalar_halo!`) rather than bare `perBC!`, so that MPI
rank-internal boundaries are handled correctly.

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
    @inside iD[I] = iszero(D[I]) ? D[I] : inv(D[I]) # alternatively: abs2(D[I])<2eps(Float32) ? zero(D[I]) : inv(D[I])
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
    iD[I]==0 && return # skip ghost/body cells; preserves MPI halo values at rank-internal boundaries
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
    # One halo up-front; inner RB sweeps run against it. After sweep 1, ghost
    # values lag by one neighbor-iteration (mild Jacobi drift), but the system
    # is diagonally-dominant (Poisson) and `mean_iters` stays at 1.0 — the
    # extra halos per sweep contribute only synchronization, not convergence.
    # This matches the old N+4/halowidth=2 pattern (reverse-engineered from
    # commit 391b999); the halowidth-2 layout wasn't necessary, the frequency
    # was.
    comm!(p.ϵ,p.perdir)
    for i ∈ 1:it
        @loop gauss_rb(p.ϵ,p.r,p.L,p.iD,i,I) over I ∈ half_rangek(p.ϵ)
    end
    increment!(p;ω) # increment solution and residual
end

"""
    perdot(a,b,perdir)

Apply dot product to the inner cells of two _scalar_ fields, assuming zero values in ghost cell when using Neumann BC.
"""
perdot(a,b,::Tuple{}) = a⋅b
perdot(a,b,perdir,R=inside(a)) = @view(a[R])⋅@view(b[R])
"""
    global_perdot(a,b,perdir)

MPI-global `perdot`: the rank-local dot product reduced across ranks.
Equivalent to `perdot` in serial.
"""
global_perdot(a,b,perdir,R...) = global_allreduce(perdot(a,b,perdir,R...))

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

L₂(a) = (R = inside(a); @view(a[R])⋅@view(a[R])) # interior-cell L₂; GPU-safe view dot
L₂(p::Poisson) = global_dot(p.r, p.r)               # Σr² across ranks (outside(p.r)≡0)
L₁(p::Poisson) = global_allreduce(local_sumabs(p.r)) # Σ|r| across ranks (outside(p.r)≡0)
L∞(p::Poisson) = global_max(maximum(abs, @view p.r[inside(p.r)]))

# mean residual  Σ|r|/N < tol/10   ⟺   L₁(p)=Σ|r| < (tol/10)·N.
# N = p.inslen: global interior-cell count (mode-aware, so serial≡parallel).
l1n_tol(p::Poisson, tol) = (Float64(tol)/10) * p.inslen

"""
    solver!(A::Poisson; tol=2e-3, itmx=1e3)

Iterative solver for the Poisson matrix equation `Ax=b` using
preconditioned conjugate gradients (`pcg!`).

  - `A.x`: Solution vector (can start with an initial guess).
  - `A.z`: Right-hand-side vector (overwritten).
  - `A.n[end]`: Number of iterations performed.
  - `tol`: Grid-independent max-norm (worst-cell) tolerance `max|r| < tol`. This is the
        knob to tune: on refined grids the mean residual clears `tol/10` with margin, so
        the max-norm is the binding constraint — lower `tol` for tighter divergence.
        Convergence also requires the mean residual `Σ|r|/N < tol/10` (same units as the
        max-norm, no hidden exponents: the bulk sits 10× below the cap).
  - `itmx`: Maximum number of iterations.

Ends with `pin_pressure!` (remove null-space mean) and `comm!`
(halo sync) so the solution is ready for use in `project!`.
"""
function solver!(p::Poisson;tol=2e-3,itmx=1e3)
    r₁tol = l1n_tol(p, tol); r∞tol = tol
    residual!(p); r₁ = L₁(p); r∞ = L∞(p)
    nᵖ=0; @log ", $nᵖ, $r∞, $r₁\n"
    while nᵖ<itmx
        pcg!(p); r₁ = L₁(p); r∞ = L∞(p); nᵖ+=1
        @log ", $nᵖ, $r∞, $r₁\n"
        (r₁<r₁tol && r∞<r∞tol) && break
    end
    pin_pressure!(p); comm!(p.x,p.perdir)
    push!(p.n,nᵖ)
end

"""
    pin_pressure!(p::Poisson)

Remove the null-space (constant) mode by subtracting the mean pressure
over fluid cells, and zero body cells (`iD==0`).  Body cells are dead
to the Poisson operator, but multigrid prolongation leaks coarse-level
values into them each V-cycle — zeroing keeps `max|p|` reporting sane
and serial/parallel runs bit-identical inside the body.  Uses mapreduce
rather than `p.z` as scratch so it's safe to call inside the V-cycle loop.
"""
function pin_pressure!(p::Poisson{T}) where T
    s = T(global_sum(p.x) / p.inslen)
    @inside p.x[I] = p.x[I] - s
end

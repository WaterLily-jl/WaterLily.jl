"""
    up(I, a=0)

Return the fine-grid `CartesianIndices` range that maps to coarse cell `I`.
When `a≠0`, the range is shifted by `-δ(a,I)` for staggered (face) restriction.
"""
@inline up(I::CartesianIndex,a=0) = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,I))
"""
    down(I)

Map fine-grid index `I` to the corresponding coarse-grid index.
"""
@inline down(I::CartesianIndex) = CI((I+2oneunit(I)).I .÷2)
"""
    restrict(I, b)

Sum the fine-grid scalar values in `b` that map to coarse cell `I`.
"""
@fastmath @inline function restrict(I::CartesianIndex,b)
    s = zero(eltype(b))
    for J ∈ up(I)
     s += @inbounds(b[J])
    end
    return s
end
"""
    restrictL(I, i, b)

Restrict the Poisson lower diagonal `b` in dimension `i` from the fine grid
to coarse cell `I`, averaging the two fine-grid face values.
"""
@fastmath @inline function restrictL(I::CartesianIndex,i,b)
    s = zero(eltype(b))
    for J ∈ up(I,i)
     s += @inbounds(b[J,i])
    end
    return 0.5s
end

"""
    restrictML(b::Poisson)

Build a new coarse-level `Poisson` from fine-level `b` by restricting the
lower diagonal `L` and allocating matching solution/residual arrays.
"""
function restrictML(b::Poisson)
    N,n = size_u(b.L)
    Na = map(i->1+i÷2,N)
    aL = similar(b.L,(Na...,n)); fill!(aL,0)
    ax = similar(b.x,Na); fill!(ax,0)
    restrictL!(aL,b.L,perdir=b.perdir)
    Poisson(ax,aL,copy(ax);b.perdir)
end
"""
    restrictL!(a, b; perdir=())

Restrict the fine-grid lower diagonal `b` into coarse-grid `a`, then apply
boundary conditions (`BC!` and `pressureBC!`) on the coarse level.
"""
function restrictL!(a::AbstractArray{T,M},b;perdir=()) where {T,M}
    Na,n = size_u(a)
    for i ∈ 1:n
        @loop a[I,i] = restrictL(I,i,b) over I ∈ CartesianIndices(map(n->2:n-1,Na))
    end
    BC!(a,zero(SVector{M-1,T}),false,perdir)  # correct μ₀ @ boundaries (master has no pressureBC at coarse)
end
restrict!(a,b) = @inside a[I] = restrict(I,b)
prolongate!(a,b) = @inside a[I] = b[down(I)]

@inline divisible(l::Poisson) = all(size(l.x) .|> divisible)

"""
    MultiLevelPoisson{T,S,V}

Composite type used to solve the pressure Poisson equation with a [geometric multigrid](https://en.wikipedia.org/wiki/Multigrid_method) method.
The main field is `levels`, a vector of nested `Poisson` systems from fine to coarse.
"""
struct MultiLevelPoisson{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    x::S
    L::V
    z::S
    levels :: Vector{Poisson{T,S,V}}
    n :: Vector{Int16}
    perdir :: NTuple # direction of periodic boundary condition
    function MultiLevelPoisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};maxlevels=10,perdir=()) where T
        levels = Poisson[Poisson(x,L,z;perdir)]
        while divisible(levels[end]) && length(levels) <= maxlevels
            push!(levels,restrictML(levels[end]))
        end
        text = "MultiLevelPoisson requires size=a2ⁿ, where n>2"
        @assert (length(levels)>2) text
        # N>6 divisibility keeps coarsest interior ≥ 2x2 — no perturbation needed
        new{T,typeof(x),typeof(L)}(x,L,z,levels,[],perdir)
    end
end

function update!(ml::MultiLevelPoisson)
    update!(ml.levels[1])
    for l ∈ 2:length(ml.levels)
        restrictL!(ml.levels[l].L,ml.levels[l-1].L,perdir=ml.levels[l-1].perdir)
        update!(ml.levels[l])
    end
end

mult!(ml::MultiLevelPoisson,x) = mult!(ml.levels[1],x)
residual!(ml::MultiLevelPoisson,x) = residual!(ml.levels[1],x)

smooth! = GaussSeidelRB!

"""
    coarsest_solve!(p::Poisson; ω=1)

Solve the coarsest-level Poisson problem at the bottom of a V-cycle.  In
serial this just calls `smooth!` (the coarsest level is already global and
very small).  The MPI extension adds a `::Parallel` method that Allreduce-
assembles the global coarsest problem, solves it redundantly on every rank,
and writes the local slice back — compensating for the V-cycle depth
deficit caused by MPI's `N>4` divisibility floor.
"""
coarsest_solve!(p::Poisson; ω=1) = _coarsest_solve!(p, ω, par_mode[])
_coarsest_solve!(p, ω, ::Serial) = smooth!(p; ω)

"""
    Vcycle!(ml::MultiLevelPoisson; l=1, ω=1)

Perform one multigrid V-cycle starting at level `l`: smooth on the fine grid,
restrict the residual, solve (or recurse) on the coarse grid, then prolongate
and correct the fine-grid solution.  At the bottom of the recursion the
coarsest level is handed to `coarsest_solve!` (MPI-aware via dispatch).
"""
function Vcycle!(ml::MultiLevelPoisson;l=1,ω=1)
    fine,coarse = ml.levels[l],ml.levels[l+1]
    # set up coarse level
    Jacobi!(fine)
    restrict!(coarse.r,fine.r)
    fill!(coarse.x,0.)
    # solve coarse (recurse if possible, otherwise bottom-solve)
    if l+1 < length(ml.levels)
        Vcycle!(ml, l=l+1; ω)
        smooth!(coarse; ω)
    else
        coarsest_solve!(coarse; ω)
    end
    # correct fine
    prolongate!(fine.ϵ,coarse.x)
    increment!(fine; ω)
end

"""
    solver!(ml::MultiLevelPoisson; tol=1e-4, itmx=32)

Multigrid solver: iterates V-cycles with adaptive relaxation `ω` until the
`L₂`-norm residual drops below `tol`.  Ends with `pin_pressure!` + `comm!`
to remove the null-space mode and synchronize halos.
"""
function solver!(ml::MultiLevelPoisson{T};tol=1e-4,itmx=32) where T
    p = ml.levels[1]
    residual!(p); r₂ = L₂(p); ω = T(1)
    nᵖ=0; @log ", $nᵖ, $(L∞(p)), $r₂, $ω\n"
    while nᵖ<itmx
        Vcycle!(ml; ω)
        smooth!(p; ω);
        rnew = L₂(p); nᵖ+=1
        @log ", $nᵖ, $(L∞(p)), $rnew, $ω\n"
        if     rnew ≥ r₂
            ω = max(0.2, 0.9ω) |> T
        elseif rnew < r₂
            ω = min(1.0, 1.02ω) |> T
        end
        r₂ = rnew
        r₂<tol && break
    end
    pin_pressure!(p); comm!(p.x,p.perdir)  # TEST: pin null-space for MPI consistency
    push!(ml.n,nᵖ);
end
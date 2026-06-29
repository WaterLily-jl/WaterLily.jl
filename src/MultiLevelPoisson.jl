# Full-coarsening (2× in every direction) maps between fine and coarse cells.
@inline up(I::CartesianIndex,a=0) = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,I))
@inline down(I::CartesianIndex) = CI((I+2oneunit(I)).I .÷2)

# Semi-coarsening up and down. `c[j]` flags which directions are coarsened (2×) between two levels
@inline up(I::CartesianIndex{n},c::NTuple{n,Bool}) where n = CartesianIndices(ntuple(j-> c[j] ? (2I.I[j]-2:2I.I[j]-1) : (I.I[j]:I.I[j]), n))
@inline down(I::CartesianIndex{n},c::NTuple{n,Bool}) where n = CI(ntuple(j-> c[j] ? (I.I[j]+2)÷2 : I.I[j], n))
# Fine faces (normal i) composing a coarse face, given the coarsening mask `c`.
@inline upL(I::CartesianIndex{n},i,c::NTuple{n,Bool}) where n =
    CartesianIndices(ntuple(j-> j==i ? (c[i] ? (2I.I[i]-2:2I.I[i]-2) : (I.I[i]:I.I[i])) :
                                       (c[j] ? (2I.I[j]-2:2I.I[j]-1) : (I.I[j]:I.I[j])), n))

@fastmath @inline function restrict(I::CartesianIndex,b,c)
    s = zero(eltype(b))
    for J ∈ up(I,c)
        s += @inbounds(b[J])
    end
    return s
end
@fastmath @inline function restrictL(I::CartesianIndex,i,b,c)
    s = zero(eltype(b))
    for J ∈ upL(I,i,c)
        s += @inbounds(b[J,i])
    end
    return c[i] ? s/2 : s  # halve only if the face-normal direction is coarsened
end

# coarsening mask: coarsen every direction that is still divisible
@inline coarsen_mask(N::NTuple) = map(divisible,N)
# mask used to build level `coarse` from `fine` (recovered from their sizes)
@inline coarsen_mask(fine,coarse) = ntuple(j-> size(coarse,j)<size(fine,j), ndims(fine))

function restrictML(b::Poisson)
    N,n = size_u(b.L)
    c = coarsen_mask(N)
    Na = ntuple(j-> c[j] ? 1+N[j]÷2 : N[j], n)
    aL = similar(b.L,(Na...,n)); fill!(aL,0)
    ax = similar(b.x,Na); fill!(ax,0)
    restrictL!(aL,b.L,c,perdir=b.perdir)
    Poisson(ax,aL,copy(ax);b.perdir)
end
function restrictL!(a::AbstractArray{T,M},b,c;perdir=()) where {T,M}
    Na,n = size_u(a)
    for i ∈ 1:n
        @loop a[I,i] = restrictL(I,i,b,c) over I ∈ CartesianIndices(map(n->2:n-1,Na))
    end
    BC!(a,zero(SVector{M-1,T}),false,perdir)  # correct μ₀ @ boundaries
end
restrict!(a,b,c) = @inside a[I] = restrict(I,b,c)
prolongate!(a,b,c) = @inside a[I] = b[down(I,c)]

@inline divisible(N::Integer) = mod(N,2)==0 && N>4
# keep coarsening while ANY direction is still divisible (semi-coarsening)
@inline divisible(l::Poisson) = any(size(l.x) .|> divisible)
"""
    MultiLevelPoisson{N,M}

Composite type used to solve the pressure Poisson equation with a [geometric multigrid](https://en.wikipedia.org/wiki/Multigrid_method) method.
The only variable is `levels`, a vector of nested `Poisson` systems.
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
        new{T,typeof(x),typeof(L)}(x,L,z,levels,[],perdir)
    end
end

function update!(ml::MultiLevelPoisson)
    update!(ml.levels[1])
    for l ∈ 2:length(ml.levels)
        c = coarsen_mask(ml.levels[l-1].x,ml.levels[l].x)
        restrictL!(ml.levels[l].L,ml.levels[l-1].L,c,perdir=ml.levels[l-1].perdir)
        update!(ml.levels[l])
    end
end

function Vcycle!(ml::MultiLevelPoisson;l=1,ω=1)
    fine,coarse = ml.levels[l],ml.levels[l+1]
    c = coarsen_mask(fine.x,coarse.x)
    # set up coarse level
    Jacobi!(fine)
    restrict!(coarse.r,fine.r,c)
    fill!(coarse.x,0)
    # solve coarse (with recursion if possible)
    l+1<length(ml.levels) && Vcycle!(ml,l=l+1; ω)
    smooth!(coarse;ω)
    # correct fine
    prolongate!(fine.ϵ,coarse.x,c)
    increment!(fine; ω)
end

mult!(ml::MultiLevelPoisson,x) = mult!(ml.levels[1],x)
residual!(ml::MultiLevelPoisson,x) = residual!(ml.levels[1],x)

smooth! = GaussSeidelRB!

function solver!(ml::MultiLevelPoisson{T};tol=2e-3,itmx=32) where T
    p = ml.levels[1]
    r₂tol = l2n_tol(p, tol); r∞tol = tol
    residual!(p); r₂ = L₂(p); r∞ = L∞(p); ω = T(1)
    nᵖ=0; @log ", $nᵖ, $r∞, $r₂, $ω\n"
    while nᵖ<itmx
        Vcycle!(ml; ω)
        smooth!(p; ω);
        rnew = L₂(p); r∞ = L∞(p); nᵖ+=1
        @log ", $nᵖ, $r∞, $rnew, $ω\n"
        if     rnew ≥ r₂
            ω = max(0.2, 0.9ω) |> T
        elseif rnew < r₂
            ω = min(1.0, 1.02ω) |> T
        end
        r₂ = rnew
        (r₂<r₂tol && r∞<r∞tol) && break
    end
    perBC!(p.x,p.perdir)
    push!(ml.n,nᵖ);
end
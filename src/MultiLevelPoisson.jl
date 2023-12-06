@inline up(I::CartesianIndex,a=0) = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,I))
@inline down(I::CartesianIndex) = CI((I+2oneunit(I)).I .÷2)
@fastmath @inline function restrict(I::CartesianIndex,b)
    s = zero(eltype(b))
    for J ∈ up(I)
     s += @inbounds(b[J])
    end
    return s
end
@fastmath @inline function restrictL(I::CartesianIndex,i,b)
    s = zero(eltype(b))
    for J ∈ up(I,i)
     s += @inbounds(b[J,i])
    end
    return 0.5s
end

function restrictML(b::Poisson)
    N,n = size_u(b.L)
    Na = map(i->1+i÷2,N)
    aL = similar(b.L,(Na...,n)); fill!(aL,0)
    ax = similar(b.x,Na); fill!(ax,0)
    restrictL!(aL,b.L,perdir=b.perdir)
    Poisson(ax,aL,copy(ax);b.perdir)
end
function restrictL!(a,b;perdir=(0,))
    Na,n = size_u(a)
    for i ∈ 1:n
        @loop a[I,i] = restrictL(I,i,b) over I ∈ CartesianIndices(map(n->2:n-1,Na))
    end
    BC!(a,zero(n);Dirichlet=false,perdir=perdir)  # correct μ₀ @ boundaries
end
restrict!(a,b) = @inside a[I] = restrict(I,b)
prolongate!(a,b) = @inside a[I] = b[down(I)]

@inline divisible(N) = mod(N,2)==0 && N>4
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
    res :: Vector{T}
    res0:: Vector{T}
    perdir :: NTuple # direction of periodic boundary condition
    function MultiLevelPoisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};maxlevels=4,perdir=(0,)) where T
        levels = Poisson[Poisson(x,L,z;perdir)]
        while all(size(levels[end].x) .|> divisible) && length(levels) <= maxlevels
            push!(levels,restrictML(levels[end]))
        end
        text = "MultiLevelPoisson requires size=a2ⁿ, where n>2"
        @assert (length(levels)>2) text
        new{T,typeof(x),typeof(L)}(x,L,z,levels,[],[],[],perdir)
    end
end
function update!(ml::MultiLevelPoisson)
    update!(ml.levels[1])
    for l ∈ 2:length(ml.levels)
        restrictL!(ml.levels[l].L,ml.levels[l-1].L,perdir=ml.levels[l-1].perdir)
        update!(ml.levels[l])
    end
end

function Vcycle!(ml::MultiLevelPoisson;l=1)
    fine,coarse = ml.levels[l],ml.levels[l+1]
    # set up coarse level
    Jacobi!(fine)
    restrict!(coarse.r,fine.r)
    fill!(coarse.x,0.)
    # solve coarse (with recursion if possible)
    l+1<length(ml.levels) && Vcycle!(ml,l=l+1)
    smooth!(coarse)
    # correct fine
    prolongate!(fine.ϵ,coarse.x)
    BC!(fine.ϵ;perdir=fine.perdir)
    increment!(fine)
end

mult!(ml::MultiLevelPoisson,x) = mult!(ml.levels[1],x)
residual!(ml::MultiLevelPoisson,x) = residual!(ml.levels[1],x)

function solver!(ml::MultiLevelPoisson;log=false,tol=1e-6,itmx=64)
    p = ml.levels[1]
    BC!(p.x;perdir=p.perdir)
    residual!(p); r₂ = L₂(p)
    push!(ml.res0,r₂)
    log && (res = [r₂])
    nᵖ=0
    while (r₂>tol || nᵖ==0) && nᵖ<itmx
        Vcycle!(ml)
        smooth!(p); r₂ = L₂(p)
        log && push!(res,r₂)
        nᵖ+=1
    end
    BC!(p.x;perdir=p.perdir)
    push!(ml.n,nᵖ); push!(ml.res,r₂)
    log && return res
end

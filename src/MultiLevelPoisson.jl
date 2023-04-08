@inline up(I::CartesianIndex,a=0) = (2I-oneunit(I)):(2I-δ(a,I))
@inline down(I::CartesianIndex) = CI((I+oneunit(I)).I .÷2)
@fastmath @inline restrict(I::CartesianIndex,b) = sum(@inbounds(b[J]) for J ∈ up(I))
@fastmath @inline restrictL(I::CartesianIndex,i,b) = 0.5sum(@inbounds(b[J,i]) for J ∈ up(I,i))

function restrictML(b::Poisson)
    N,n = size_u(b.L)
    Na = map(i->1+i÷2,N)
    aL = similar(b.L,(Na...,n)) |> OA(n); fill!(aL,0)
    ax = similar(b.x,Na) |> OA(); fill!(ax,0)
    restrictL!(aL,b.L)
    Poisson(ax,aL)
end
function restrictL!(a,b)
    Na,n = size_u(a)
    for i ∈ 1:n
        @loop a[I,i] = restrictL(I,i,b) over I ∈ Na.-2
    end
end
restrict!(a,b) = @inside a[I] = restrict(I,b)
prolongate!(a,b) = @inside a[I] = b[down(I)]

@inline divisible(N) = mod(N,2)==0 && N>4
"""
    MultiLevelPoisson{N,M}

Composite type used to solve the pressure Poisson equation with a [geometric multigrid](https://en.wikipedia.org/wiki/Multigrid_method) method.
The only variable is `levels`, a vector of nested `Poisson` systems.
"""
struct MultiLevelPoisson{T,S,V} <: AbstractPoisson{T,S,V}
    levels :: Vector{Poisson{T,S,V}}
    n :: Vector{Int16}
    function MultiLevelPoisson(x::AbstractArray{T},L::AbstractArray{T}) where T
        @assert all(first.(axes(x)).==0) # up/down require fixed indexing
        levels = Poisson[Poisson(x,L)]
        while all(size(levels[end].x) .|> divisible)
            push!(levels,restrictML(levels[end]))
        end
        text = "MultiLevelPoisson requires size=a2ⁿ, where a<31, n>2"
        @assert (length(levels)>2 && all(size(levels[end].x).<31)) text
        new{T,typeof(x),typeof(L)}(levels,[])
    end
end
function update!(ml::MultiLevelPoisson)
    update!(ml.levels[1])
    for l ∈ 2:length(ml.levels)
        restrictL!(ml.levels[l].L,ml.levels[l-1].L)
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
    SOR!(coarse);SOR!(coarse);SOR!(coarse)
    # correct fine
    prolongate!(fine.ϵ,coarse.x)
    increment!(fine)
end

mult(ml::MultiLevelPoisson,x) = mult(ml.levels[1],x)

function solver!(ml::MultiLevelPoisson,b;log=false,tol=1e-3,itmx=32)
    p = ml.levels[1]
    @assert axes(p.x)==axes(b)
    residual!(p,b); r₂ = L₂(p.r)
    log && (res = [r₂])
    nᵖ=0
    while r₂>tol && nᵖ<itmx
        Vcycle!(ml)
        SOR!(p);SOR!(p);SOR!(p); r₂ = L₂(p.r)
        log && push!(res,r₂)
        nᵖ+=1
    end
    push!(ml.n,nᵖ)
    log && return res
end

@inline near(I::CartesianIndex,a=0) = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,I))

@fastmath function restrict(b::Array{Float64,m}) where m
    N = ntuple(i-> i==m ? m-1 : 1+size(b,i)÷2, m)
    a = zeros(N)
    for i ∈ 1:m-1, I ∈ inside(N[1:m-1])
        @inbounds a[I,i] = 0.5sum(b[J,i] for J ∈ near(I,i))
    end
    PoissonSys(a)
end

@inline divisible(N) = mod(N,2)==0 && N>4

struct MultiLevelPS{N,M} <: Poisson{N,M}
    levels :: Vector{PoissonSys{N,M}}
    function MultiLevelPS(L::Array{Float64,n}) where n
        levels = Vector{PoissonSys}()
        push!(levels,PoissonSys(L))
        while all(size(levels[end].x) .|> divisible)
            push!(levels,restrict(levels[end].L))
        end
        @assert all(size(levels[end].x).<10) "GMG requires size=a2ⁿ, where a<10"
        new{n,n-1}(levels)
    end
end

@fastmath restrict!(a::Array{Float64},b::Array{Float64}) = @simd for I ∈ inside(a)
    @inbounds a[I] = sum(b[J] for J ∈ near(I))
end

prolongate!(a::Array{Float64},b::Array{Float64}) = for I ∈ inside(b)
    for J ∈ near(I)
        @inbounds a[J] = b[I]
end;end

function Vcycle!(p::MultiLevelPS;l=1)
    # set up level l+1
    fill!(p.levels[l+1].x,0.)
    GS!(p.levels[l],it=0)
    restrict!(p.levels[l+1].r,p.levels[l].r)
    # recurse
    l+1<length(p.levels) && Vcycle!(p,l=l+1)
    # correct level l
    GS!(p.levels[l+1],it=4)
    prolongate!(p.levels[l].ϵ,p.levels[l+1].x)
    increment!(p.levels[l])
end

function solve!(x::Array{Float64,m},p::MultiLevelPS{n,m},b::Array{Float64,m};log=false,tol=1e-4) where {n,m}
    p1 = p.levels[1]; p1.x .= x
    residual!(p1,b); r₂ = L₂(p1.r)
    log && (res = [r₂])
    while r₂>tol
        Vcycle!(p)
        GS!(p1,it=4); r₂ = L₂(p1.r)
        log && push!(res,r₂)
    end
    x .= p1.x
    return log ? res : nothing
end

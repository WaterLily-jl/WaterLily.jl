include("PoissonSys.jl")

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

struct MultiLevelPS{N,M}
    levels :: Vector{PoissonSys{N,M}}
    function MultiLevelPS(L::Array{Float64,n}) where n
        levels = Vector{PoissonSys}()
        push!(levels,PoissonSys(L))
        while all(size(levels[end].x) .|> divisible)
            push!(levels,restrict(levels[end].L))
        end
        new{n,n-1}(levels)
    end
end

@fastmath restrict!(a::Array{Float64},b::Array{Float64}) =
@simd for I ∈ inside(a)
    @inbounds a[I] = sum(b[J] for J ∈ near(I))
end

prolongate!(a::Array{Float64},b::Array{Float64}) =
    for I ∈ inside(b); for J ∈ near(I)
        @inbounds a[J] = b[I]
end;end

@fastmath function Vcycle!(p::MultiLevelPS;l=1)
    SOR!(p.levels[l])
    l==length(p.levels) && return
    fill!(p.levels[l+1].x,0.)
    restrict!(p.levels[l].r,p.levels[l+1].r)
    Vcycle!(p,l=l+1)
    SOR!(p.levels[l+1],ω=1.8)
    prolongate!(p.levels[l].ϵ,p.levels[l+1].x)
    increment!(p.levels[l])
end

function GMG_test(n=2^4)
    c = ones(n+2,n+2,2); BC!(c,[0. 0.])
    @time p = MultiLevelPS(c)
    b = Float64[i for i∈1:n+2, j∈1:n+2]
    a = AU(size(p.levels[2].x))
    @time restrict!(a,b)
    @time prolongate!(b,a)
    return p,a,b
end

function solve!(x::Array{Float64,m},p::MultiLevelPS{n,m},b::Array{Float64,m};log=false,tol=1e-4) where {n,m}
    p1 = p.level[1]; p1.x .= x
    residual!(p1,b); r₂ = L₂(p1.r)
    log && (res = [r₂])
    while r₂>tol
        Vcycle!(p)
        SOR!(p1,ω=1.8); r₂ = L₂(p1)
        log && push!(res,r₂)
    end
    x .= p1.x
    return log ? (x,res) : x
end

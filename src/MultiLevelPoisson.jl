@inline near(I::CartesianIndex,a=0) = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,I))

@fastmath function restrictML(b::Array{Float64,m}) where m
    N = ntuple(i-> i==m ? m-1 : 1+size(b,i)÷2, m)
    a = zeros(N)
    @inbounds for i ∈ 1:m-1, I ∈ inside(N[1:m-1])
        a[I,i] = 0.5sum(b[J,i] for J ∈ near(I,i))
    end
    Poisson(a)
end

@fastmath restrict!(a::Array{Float64},b::Array{Float64}) = @inbounds @simd for I ∈ inside(a)
    a[I] = sum(@inbounds(b[J]) for J ∈ near(I))
end

prolongate!(a::Array{Float64},b::Array{Float64}) = @inbounds for I ∈ inside(b)
    @simd for J ∈ near(I)
        a[J] = b[I]
end;end

@inline divisible(N) = mod(N,2)==0 && N>4

struct MultiLevelPoisson{N,M} <: AbstractPoisson{N,M}
    levels :: Vector{Poisson{N,M}}
    function MultiLevelPoisson(L::Array{Float64,n}) where n
        levels = [Poisson(L)]
        while all(size(levels[end].x) .|> divisible)
            push!(levels,restrictML(levels[end].L))
        end
        text = "MultiLevelPoisson requires size=a2ⁿ, where a<10, n>1"
        @assert length(levels)>1 & all(size(levels[end].x).<10) text
        new{n-1,n}(levels)
    end
end

function Vcycle!(ml::MultiLevelPoisson;l=1)
    fine,coarse = ml.levels[l],ml.levels[l+1]
    # set up coarse level
    GS!(fine,it=0)
    restrict!(coarse.r,fine.r)
    fill!(coarse.x,0.)
    # solve coarse (with recursion if possible)
    l+1<length(ml.levels) && Vcycle!(ml,l=l+1)
    GS!(coarse,it=2)
    # correct fine
    prolongate!(fine.ϵ,coarse.x)
    increment!(fine)
end

mult(ml::MultiLevelPoisson,x) = mult(ml.levels[1],x)

function solve!(x::Array{Float64,n},ml::MultiLevelPoisson{n},b::Array{Float64,n};log=false,tol=1e-3,itmx=32) where n
    p = ml.levels[1]
    p.x .= x
    residual!(p,b); r₂ = L₂(p.r)
    log && (res = [r₂])
    nᵖ=0
    while r₂>tol && nᵖ<itmx
        Vcycle!(ml)
        GS!(p,it=2); r₂ = L₂(p.r)
        5tol>r₂>tol && (GS!(p,it=2); r₂ = L₂(p.r))
        log && push!(res,r₂)
        nᵖ+=1
    end
    x .= p.x
    return log ? res : nᵖ
end

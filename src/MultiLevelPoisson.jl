@inline near(I::CartesianIndex,a=0) = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,I))

function restrictML(b::Array{T,m}) where {T,m}
    N = ntuple(i-> i==m ? m-1 : 1+size(b,i)÷2, m)
    a = zeros(T,N)
    restrictL!(a,b)
    Poisson(a)
end
@fastmath function restrictL!(a::Array{T,m},b::Array{T,m}) where {T,m}
    @inbounds for i ∈ 1:m-1, I ∈ inside(size(a)[1:m-1])
        a[I,i] = 0.5sum(@inbounds(b[J,i]) for J ∈ near(I,i))
    end
end

@fastmath restrict!(a::Array,b::Array) = @inbounds @simd for I ∈ inside(a)
    a[I] = sum(@inbounds(b[J]) for J ∈ near(I))
end

prolongate!(a::Array,b::Array) = @inbounds for I ∈ inside(b)
    @simd for J ∈ near(I)
        a[J] = b[I]
end;end

@inline divisible(N) = mod(N,2)==0 && N>4

struct MultiLevelPoisson{N,M,T} <: AbstractPoisson{N,M,T}
    levels :: Vector{Poisson{N,M,T}}
    n :: Vector{Int16}
    function MultiLevelPoisson(L::Array{T,n}) where {T,n}
        levels = [Poisson(L)]
        while all(size(levels[end].x) .|> divisible)
            push!(levels,restrictML(levels[end].L))
        end
        text = "MultiLevelPoisson requires size=a2ⁿ, where a<31, n>2"
        @assert (length(levels)>2 && all(size(levels[end].x).<31)) text
        new{n-1,n,T}(levels,[])
    end
end
function update!(ml::MultiLevelPoisson,L::Array)
    update!(ml.levels[1],L)
    for l ∈ 2:length(ml.levels)
        restrictL!(ml.levels[l].L,ml.levels[l-1].L)
        set_diag!(ml.levels[l])
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

function solver!(x::Array,ml::MultiLevelPoisson,b::Array;log=false,tol=1e-3,itmx=32)
    p = ml.levels[1]
    @assert size(p.x)==size(x)
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
    push!(ml.n,nᵖ)
    log && return res
end

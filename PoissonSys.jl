struct PoissonSys{N,M}
    L :: Array{Float64,N} # Lower diagonal coefficients
    D :: Array{Float64,M} # Diagonal coefficients
    iD :: Array{Float64,M} # 1/Diagonal
    x :: Array{Float64,M} # solution
    r :: Array{Float64,M} # residual
    function PoissonSys(L::Array{Float64,n}) where n
        N = size(L); M = N[1:end-1]; m = length(M)
        @assert N[end] == m
        x,r,D,iD = AU(M),AU(M),AU(M),AU(M)
        for I ∈ inside(M)
            D[I] = -sum(i->L[I,i]+L[I+δ(i,m),i],1:m)
            iD[I] = abs2(D[I])<1e-8 ? 0. : inv(D[I])
        end
        new{n,m}(L,D,iD,x,r)
    end
end

@fastmath @inline function multLU(I::CartesianIndex{d},L,x) where d
    s = 0
    for i ∈ 1:d
        @inbounds s += x[I-δ(i,I)]*L[I,i]+x[I+δ(i,I)]*L[I+δ(i,I),i]
    end
    return s
end
@fastmath @inline mult(I,L,D,x) = multLU(I,L,x)+x[I]*D[I]
mult!(p::PoissonSys{n,m}) where {n,m} = @simd for I ∈ inside(p.r)
    @inbounds p.r[I] = mult(I,p.L,p.D,p.x)
end

@fastmath function resid(p::PoissonSys{n,m}) where {n,m}
    s = 0.
    @simd for I ∈ inside(p.r)
        s += abs2(p.r[I]-mult(I,p.L,p.D,p.x))
    end
    return s
end

@fastmath SOR!(p::PoissonSys{n,m}; ω::Real=1.5) where {n,m} = @simd for I ∈ inside(p.r)
    @inbounds p.x[I] += ω*((p.r[I]-multLU(I,p.L,p.x))*p.iD[I]-p.x[I])
end

function solve!(x::Array{Float64,m},p::PoissonSys{n,m},b::Array{Float64,m},log=false) where {n,m}
    p.x .= x; p.r.= b
    r = resid(p)
    log && (res = [r])
    while r>1e-4
        SOR!(p,ω=1.8); r = resid(p)
        log && push!(res,r)
    end
    x .= p.x
    return log ? (x,res) : x
end

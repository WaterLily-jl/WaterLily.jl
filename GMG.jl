function GMG(c)
    n,m,d = size(c)
    inv_diag = zeros(n,m)
    for I ∈ CR((2:n-1,2:m-1))
        diag = -(c[I,1]+c[I+δ(1,I),1]+c[I,2]+c[I+δ(2,I),2])
        inv_diag[I] = diag==0. ? diag : inv(diag)
    end
    return inv_diag
end
@fastmath @inline multLU(I,c,p) = (p[I-δ(1,I)]*c[I,1]+p[I+δ(1,I)]*c[I+δ(1,I),1]
                                  +p[I-δ(2,I)]*c[I,2]+p[I+δ(2,I)]*c[I+δ(2,I),2])

@fastmath function projectGMG!(p,u,c,σ,inv_diag,Δt)
    n,m,d = size(u)
    R = CR((2:n-1,2:m-1))
    @simd for I ∈ R
        @inbounds σ[I] = ∇(I,u)/Δt
    end
    # SOR
    ω,itmx = 1.5,10
    for it ∈ 1:itmx; @simd for I ∈ R
        @inbounds p[I] += ω*((σ[I]-multLU(I,c,p))*inv_diag[I]-p[I])
    end; end

    for a ∈ 1:d; @simd for I ∈ R
        @inbounds u[I,a] -= Δt*c[I,a]*∂(a,I,p)
    end;end
end

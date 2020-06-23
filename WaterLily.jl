using Images,Plots
show(f) = plot(Gray.(f'[end:-1:1,:]))
show(f,fmin,fmax) = show((f.-fmin)/(fmax-fmin))
show_scaled(σ) = show(σ,minimum(σ),maximum(σ))

@inline CI(a...) = CartesianIndex(a...)
@inline CR(a...) = CartesianIndices(a...)
@inline δ(a,I::CartesianIndex{N}) where {N} = CI(ntuple(i -> i==a ? 1 : 0, N))
@inline ∂(a,I,f) = @inbounds f[I]-f[I-δ(a,I)]
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])*0.5
@inline ∇(I::CartesianIndex{2},u) = u[I+δ(1,I),1]-u[I,1]+u[I+δ(2,I),2]-u[I,2]
function median(a,b,c)
    x = a-b
    if x*(b-c) ≥ 0
        return b
    elseif x*(a-c) > 0
        return c
    else
        return a
    end
end
@fastmath quick(u,c,d) = median((5c+2d-u)/6,c,median(10c-9u,c,d))
@inline ϕu(a,I,f,u) = @inbounds u>0 ? u*quick(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*quick(f[I+δ(a,I)],f[I],f[I-δ(a,I)])

@fastmath function tracer_transport!(r,f,u;Pe=0.1)
    N = size(u)
    for b ∈ 1:N[3], j ∈ 2:N[2], i ∈ 2:N[1]
        I,uᵇ = CI(i,j),u[i,j,b]
        if I[b]==2 || I[b]==N[b]
            Φ = ϕ(b,I,f)*uᵇ-Pe*∂(b,I,f)
        else
            Φ = ϕu(b,I,f,uᵇ)-Pe*∂(b,I,f)
        end
        @inbounds r[I] += Φ
        @inbounds r[I-δ(b,I)] -= Φ
    end
end

function BC!(f::Array{T,d}) where {T<:Real, d}
    for I ∈ CR(f), b ∈ 1:d
        if I[b] == 1
            f[I] = f[I+δ(b,I)]
        elseif I[b] == size(f)[b]
            f[I] = f[I-δ(b,I)]
        end
    end
end

function BC!(u::Array{T,3},U) where T<:Real
    N = size(u)
    @simd for I ∈ CR(u)
        a = I[3]; b = a%N[3]+1
        if I[a] in (1,2,N[a])
            u[I] = U[a]
        elseif I[b] == 1
            u[I] = u[I+δ(b,I)]
        elseif I[b] == N[b]
            u[I] = u[I-δ(b,I)]
        end
    end
end

using SparseArrays: sparse
function construct(cx,cy)
    # get dimensions
    n,m = size(cy,1),size(cx,2)
    N = m*n

    # compute diagonal and cat all values
    diag = @inbounds [-(cx[i,j]+cy[i,j]+cx[i+1,j]+cy[i,j+1]) for i ∈ 1:n, j ∈ 1:m]
    V = vcat(diag[:],cy[:,2:m][:],cy[:,2:m][:],cx[2:n,:][:],cx[2:n,:][:])

    # compute strides and cat all indices
    s1,s2 = setdiff(2:N,n+1:n:N),setdiff(1:N-1,n:n:N)
    I = vcat(1:N,1:n*(m-1),1+n:N,s1,s2)
    J = vcat(1:N,1+n:N,1:n*(m-1),s2,s1)

    return sparse(I,J,V) # return sparse Poisson matrix
end

using AlgebraicMultigrid: solve!,ruge_stuben,GaussSeidel
function MG(c)
    n,m,d = size(c)
    ruge_stuben(construct(c[2:n,2:m-1,1],c[2:n-1,2:m,2]),
        presmoother = GaussSeidel(iter=0)) # no presmoother
end

using LinearAlgebra: diag,norm
function project!(p,u,c,σ,x,ml)
    n,m,d = size(u)
    R = CR((2:n-1,2:m-1))
    @simd for k ∈ 1:length(R); I=R[k]
        @inbounds σ[k] = ∇(I,u)
        @inbounds x[k] = p[I]
    end
    solve!(x,ml,σ,tol=1e-3/norm(σ))
    # x,residuals = solve!(x,ml,σ,log=true,tol=1e-3/norm(σ))
    # println([length(residuals) residuals[end] 1e-4/norm(σ) norm(σ)])
    x .-= sum(x)/length(x)
    @simd for k ∈ 1:length(R); I=R[k]
        @inbounds p[I] = x[k]
    end
    for a ∈ 1:d; @simd for I ∈ R
        @inbounds u[I,a] -= c[I,a]*∂(a,I,p)
    end;end
end

function mom_transport!(r,u;ν=0.1)
    N = size(u)
    for a ∈ 1:N[3], b ∈ 1:N[3], j ∈ 2:N[2], i ∈ 2:N[1]
        Iᵃ,Iᵇ = CI(i,j,a),CI(i,j,b)
        if Iᵇ[b]==2 || Iᵇ[b]==N[b]
            Φ = ϕ(b,Iᵃ,u)*ϕ(a,Iᵇ,u)-ν*∂(b,Iᵃ,u)
        else
            Φ = ϕu(b,Iᵃ,u,ϕ(a,Iᵇ,u))-ν*∂(b,Iᵃ,u)
        end
        @inbounds r[Iᵃ] += Φ
        @inbounds r[Iᵃ-δ(b,Iᵃ)] -= Φ
    end
end

struct flow
    u;c;r
    p
    ml;σ;p_vec
end
function flow(u,c)
    n,m,d = size(u)
    flow(u,c,similar(u),zeros(n,m),
        MG(c),zeros((n-2)*(m-2)),zeros((n-2)*(m-2)))
end

@fastmath function mom_step!(a::flow;Δt=0.25,ν=0.1,U=[1. 0.])
    fill!(a.r,0.)
    mom_transport!(a.r,a.u,ν=ν)
    @. a.u += Δt*a.c*a.r; BC!(a.u,U)
    a.p .*= Δt
    project!(a.p,a.u,a.c,a.σ,a.p_vec,a.ml)
    a.p ./= Δt
    BC!(a.u,U);
end

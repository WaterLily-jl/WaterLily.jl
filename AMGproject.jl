using SparseArrays: sparse
using AlgebraicMultigrid: solve!,ruge_stuben,GaussSeidel
using LinearAlgebra: norm

function constructSparse(cx,cy)
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

function AMG(c)
    n,m,d = size(c)
    ruge_stuben(constructSparse(c[2:n,2:m-1,1],c[2:n-1,2:m,2]),
        presmoother = GaussSeidel(iter=0)) # no presmoother
end

function projectAMG!(p,u,c,σ,x,ml,Δt)
    n,m,d = size(u)
    R = CR((2:n-1,2:m-1))
    @simd for k ∈ 1:length(R); I=R[k]
        @inbounds σ[k] = ∇(I,u)/Δt
        @inbounds x[k] = p[I]
    end
    solve!(x,ml,σ,tol=1e-3/Δt/norm(σ))
    # x,residuals = solve!(x,ml,σ,log=true,tol=1e-3/Δt/norm(σ))
    # println([length(residuals) residuals[end] norm(σ)])
    x .-= sum(x)/length(x)
    @simd for k ∈ 1:length(R); I=R[k]
        @inbounds p[I] = x[k]
    end
    for a ∈ 1:d; @simd for I ∈ R
        @inbounds u[I,a] -= Δt*c[I,a]*∂(a,I,p)
    end;end
end

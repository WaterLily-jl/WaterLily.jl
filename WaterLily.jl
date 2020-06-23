using Images,Plots
show(f) = plot(Gray.(f'[end:-1:1,:]))
show(f,fmin,fmax) = show((f.-fmin)/(fmax-fmin))
show_scaled(σ) = show(σ,minimum(σ),maximum(σ))

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

@inline ∫ˣϕuⁿ(i,j,f,u) = @inbounds u>0 ? u*quick(f[i-2,j],f[i-1,j],f[i,j]) : u*quick(f[i+1,j],f[i,j],f[i-1,j])
@inline ∫ʸϕuⁿ(i,j,f,v) = @inbounds v>0 ? v*quick(f[i,j-2],f[i,j-1],f[i,j]) : v*quick(f[i,j+1],f[i,j],f[i,j-1])
@inline ∫ˣϕ(i,j,f) = @inbounds (f[i-1,j]+f[i,j])*0.5
@inline ∫ʸϕ(i,j,f) = @inbounds (f[i,j-1]+f[i,j])*0.5
@inline ∫ˣ∂x(i,j,f) = @inbounds f[i,j]-f[i-1,j]
@inline ∫ʸ∂y(i,j,f) = @inbounds f[i,j]-f[i,j-1]

@inline CI(a...) = CartesianIndex(a...)
@inline δ(a,I::CartesianIndex{N}) where {N} = CI(ntuple(i -> i==a ? 1 : 0, N))
@inline ∂(a,I,f) = @inbounds f[I]-f[I-δ(a,I)]
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])*0.5
@inline ∇(I,u) = sum(a -> u[I+δ(a,I),a]-u[I,a], 1:size(u)[end])

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
@fastmath function BCᶜ!(f)
    n,m = size(f)
    f[1,:] .= @view f[2,:]
    f[n,:] .= @view f[n-1,:]
    f[:,1] .= @view f[:,2]
    f[:,m] .= @view f[:,m-1]
    return
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
function project!(p,ux,uy,cx,cy,σ,x,ml)
    u = cat(ux,uy,dims=3)
    c = cat(cx,cy,dims=3)
    n,m = size(p)
    R = CartesianIndices((2:n-1,2:m-1))
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
    for a ∈ 1:2; @simd for I ∈ R
        @inbounds u[I,a] -= c[I,a]*∂(a,I,p)
    end;end
    ux .= u[:,:,1]; uy .= u[:,:,2]
end

@fastmath function BC!(u,U)
    n,m,d = size(u)
    u[1,2:m-1,1] .= U[1]
    u[2,2:m-1,1] .= U[1]
    u[n,2:m-1,1] .= U[1]
    u[:,1,1] .= @view u[:,2,1]
    u[:,m,1] .= @view u[:,m-1,1]
    u[2:n-1,1,2] .= U[2]
    u[2:n-1,2,2] .= U[2]
    u[2:n-1,m,2] .= U[2]
    u[1,:,2] .= @view u[2,:,2]
    u[n,:,2] .= @view u[n-1,:,2]
    return
end

@fastmath function BCˣ!(f,val)
    n,m = size(f)
    f[1,2:m-1] .= val
    f[2,2:m-1] .= val
    f[n,2:m-1] .= val
    for i ∈ 1:n
        f[i,1] = f[i,2]
        f[i,m] = f[i,m-1]
    end
    return
end
@fastmath function BCʸ!(f,val)
    n,m = size(f)
    f[2:n-1,1] .= val
    f[2:n-1,2] .= val
    f[2:n-1,m] .= val
    for j ∈ 1:m
        f[1,j] = f[2,j]
        f[n,j] = f[n-1,j]
    end
    return
end

function mom_transport!(rˣ,rʸ,uˣ,uʸ;ν=0.1)
    u = cat(uˣ,uʸ,dims=3)
    N = size(u)
    r = zeros(N)
    for a ∈ 1:N[3], b ∈ 1:N[3], j ∈ 2:N[2], i ∈ 2:N[1]
        Iᵃ,Iᵇ = CI(i,j,a),CI(i,j,b)
        if Iᵇ[b]==2 || Iᵇ[b]==N[b]
            Φ = ϕ(b,Iᵃ,u)*ϕ(a,Iᵇ,u)-ν*∂(b,Iᵃ,u)
        else
            Φ = ϕu(b,Iᵃ,u,ϕ(a,Iᵇ,u))-Pe*∂(b,Iᵃ,u)
        end
        @inbounds r[Iᵃ] += Φ
        @inbounds r[Iᵃ-δ(b,Iᵃ)] -= Φ
    end
    rˣ .= r[:,:,1]; rʸ .= r[:,:,2]
    return
end

struct flow
    uˣ;uʸ;cˣ;cʸ;rˣ;rʸ
    p
    ml;σ;p_vec
end
function flow(uˣ,uʸ,cˣ,cʸ)
    n,m = size(uˣ)
    flow(uˣ,uʸ,cˣ,cʸ,similar(uˣ),similar(uˣ),zeros(n,m),
        MG(cˣ,cʸ),zeros((n-2)*(m-2)),zeros((n-2)*(m-2)))
end

function mom_step!(a::flow;Δt=0.25,ν=0.1,Uˣ=1.,Uʸ=0.)
    fill!(a.rˣ,0.); fill!(a.rʸ,0.);
    mom_transport!(a.rˣ,a.rʸ,a.uˣ,a.uʸ,ν=ν)
    a.uˣ .+= Δt.*a.cˣ.*a.rˣ; BCˣ!(a.uˣ,Uˣ)
    a.uʸ .+= Δt.*a.cʸ.*a.rʸ; BCʸ!(a.uʸ,Uʸ)
    a.p .*= Δt
    project!(a.p,a.uˣ,a.uʸ,a.cˣ,a.cʸ,a.σ,a.p_vec,a.ml)
    a.p ./= Δt
    BCˣ!(a.uˣ,Uˣ); BCʸ!(a.uʸ,Uʸ)
end

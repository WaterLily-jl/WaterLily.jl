using Images,Plots
show(f) = plot(Gray.(f'[end:-1:1,:]))
show(f,fmin,fmax) = show((f.-fmin)/(fmax-fmin))
show_scaled(σ) = show((σ.-minimum(σ))./(maximum(σ)-minimum(σ)))

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
@inline ∫ˣϕuⁿ(i,j,f,u) = @inbounds u>0 ? u*quick(f[i-2,j],f[i-1,j],f[i,j]) : u*quick(f[i+1,j],f[i,j],f[i-1,j])
@inline ∫ʸϕuⁿ(i,j,f,v) = @inbounds v>0 ? v*quick(f[i,j-2],f[i,j-1],f[i,j]) : v*quick(f[i,j+1],f[i,j],f[i,j-1])
@inline ∫ˣϕ(i,j,f) = @inbounds (f[i-1,j]+f[i,j])*0.5
@inline ∫ʸϕ(i,j,f) = @inbounds (f[i,j-1]+f[i,j])*0.5
@inline ∫ˣ∂x(i,j,f) = @inbounds f[i,j]-f[i-1,j]
@inline ∫ʸ∂y(i,j,f) = @inbounds f[i,j]-f[i,j-1]

@fastmath function transport!(f,f₀,u,v;Δt=0.25,Pe=0.1)
    n,m = size(f)
    for i ∈ 2:n, j ∈ 2:m
        if i==2 || i==n
            Φx = ∫ˣϕ(i,j,f₀)*u[i,j]-Pe*∫ˣ∂x(i,j,f₀)
        else
            Φx = ∫ˣϕuⁿ(i,j,f₀,u[i,j])-Pe*∫ˣ∂x(i,j,f₀)
        end
        if j==2 || j==m
            Φy = ∫ʸϕ(i,j,f₀)*v[i,j]-Pe*∫ʸ∂y(i,j,f₀)
        else
            Φy = ∫ʸϕuⁿ(i,j,f₀,v[i,j])-Pe*∫ʸ∂y(i,j,f₀)
        end
        @inbounds f[i,j] += Δt*(Φx+Φy)
        @inbounds f[i-1,j] -= Δt*Φx
        @inbounds f[i,j-1] -= Δt*Φy
    end
end

using SparseArrays: sparse
using AlgebraicMultigrid: solve!,ruge_stuben

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

function MG(cx,cy)
    n,m = size(cx)
    ruge_stuben(construct(cx[2:n,2:m-1],cy[2:n-1,2:m]))
end

using LinearAlgebra: diag
function project!(p,ux,uy,cx,cy,ml)
    n,m = size(p)
    σ = [@inbounds ∫ˣ∂x(i+1,j,ux)+∫ʸ∂y(i,j+1,uy) for i ∈ 2:n-1, j ∈ 2:m-1][:]
    x = @inbounds p[2:n-1,2:m-1][:]
    solve!(x,ml,σ)
    x[diag(ml.levels[1].A).==0.] .= 0
    @inbounds p[2:n-1,2:m-1] .= reshape(x,n-2,m-2).-sum(x)/length(x)
    for i ∈ 3:n-1, j ∈ 2:m-1
        @inbounds ux[i,j] -= cx[i,j]*∫ˣ∂x(i,j,p)
    end
    for i ∈ 2:n-1, j ∈ 3:m-1
        @inbounds uy[i,j] -= cy[i,j]*∫ʸ∂y(i,j,p)
    end
end


@fastmath function BCᶜ!(f)
    n,m = size(f)
    f[1,:] .= f[2,:]
    f[n,:] .= f[n-1,:]
    f[:,1] .= f[:,2]
    f[:,m] .= f[:,m-1]
end
@fastmath function BCˣ!(f,val)
    n,m = size(f)
    f[1:2,2:m-1].= val
    f[n,2:m-1].= val
    f[:,1] .= f[:,2]
    f[:,m] .= f[:,m-1]
end
@fastmath function BCʸ!(f,val)
    n,m = size(f)
    f[2:n-1,1:2].=val
    f[2:n-1,m].=val
    f[1,:] .= f[2,:]
    f[n,:] .= f[n-1,:]
end

function mom_transport!(rˣ,rʸ,uˣ,uʸ;ν=0.1)
    n,m = size(uˣ)
    for i ∈ 2:n, j ∈ 2:m
        if i==2 || i==n
            Φˣˣ = ∫ˣϕ(i,j,uˣ)^2-ν*∫ˣ∂x(i,j,uˣ)
            Φʸˣ = ∫ʸϕ(i,j,uˣ)*∫ˣϕ(i,j,uʸ)-ν*∫ˣ∂x(i,j,uʸ)
        else
            Φˣˣ = ∫ˣϕuⁿ(i,j,uˣ,∫ˣϕ(i,j,uˣ))-ν*∫ˣ∂x(i,j,uˣ)
            Φʸˣ = ∫ˣϕuⁿ(i,j,uʸ,∫ʸϕ(i,j,uˣ))-ν*∫ˣ∂x(i,j,uʸ)
        end
        if j==2 || j==m
            Φˣʸ = ∫ʸϕ(i,j,uˣ)*∫ˣϕ(i,j,uʸ)-ν*∫ʸ∂y(i,j,uˣ)
            Φʸʸ = ∫ʸϕ(i,j,uʸ)^2-ν*∫ʸ∂y(i,j,uʸ)
        else
            Φˣʸ = ∫ʸϕuⁿ(i,j,uˣ,∫ˣϕ(i,j,uʸ))-ν*∫ʸ∂y(i,j,uˣ)
            Φʸʸ = ∫ʸϕuⁿ(i,j,uʸ,∫ʸϕ(i,j,uʸ))-ν*∫ʸ∂y(i,j,uʸ)
        end
        @inbounds rˣ[i,j] += Φˣˣ+Φˣʸ
        @inbounds rˣ[i-1,j] -= Φˣˣ
        @inbounds rˣ[i,j-1] -= Φˣʸ
        @inbounds rʸ[i,j] += Φʸˣ+Φʸʸ
        @inbounds rʸ[i-1,j] -= Φʸˣ
        @inbounds rʸ[i,j-1] -= Φʸʸ
    end
end

function mom_step!(p,uˣ,uʸ,cˣ,cʸ,ml;Δt=0.25,ν=0.1,Uˣ=1.,Uʸ=0.)
    rˣ = zeros(size(uˣ)); rʸ = zeros(size(uˣ))
    mom_transport!(rˣ,rʸ,uˣ,uʸ,ν=ν)
    uˣ .+= Δt.*cˣ.*rˣ; BCˣ!(uˣ,Uˣ)
    uʸ .+= Δt.*cʸ.*rʸ; BCʸ!(uʸ,Uʸ)
    p .*= Δt
    project!(p,uˣ,uʸ,cˣ,cʸ,ml)
    p ./= Δt
    BCˣ!(uˣ,Uˣ); BCʸ!(uʸ,Uʸ)
end

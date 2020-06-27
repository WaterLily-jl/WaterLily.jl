include("util.jl")

@inline ∂(a,I::CartesianIndex{d},f::Array{Float64,d}) where d = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::Array{Float64,n}) where {n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])*0.5
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

function BC!(u::Array{Float64,3},U)
    _dirichlet(u,1,CR(()),CR(1:size(u,2)),1,U[1])
    _dirichlet(u,2,CR(1:size(u,1)),CR(()),2,U[2])
    _nuemann(u,2,CR(1:size(u,1)),CR(()),1)
    _nuemann(u,1,CR(()),CR(1:size(u,2)),2)
end
function BC!(f::Array{Float64,2})
    _nuemann(f,1,CR(()),CR(1:size(f,2)),CI())
    _nuemann(f,2,CR(1:size(f,1)),CR(()),CI())
end
@inline _nuemann(f,b,left,right,a) = for r ∈ right, l ∈ left
    f[l,1,r,a] = f[l,2,r,a]; f[l,size(f,b),r,a] = f[l,size(f,b)-1,r,a]
end
@inline _dirichlet(f,b,left,right,a,F) = for r ∈ right, l ∈ left
    f[l,1,r,a] = f[l,2,r,a] = f[l,size(f,b),r,a] = F
end

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

@fastmath function mom_transport!(r,u;ν=0.1)
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

struct Flow{N,M}
    u :: Array{Float64,N} # velocity vector field
    c :: Array{Float64,N} # BDIM \mu_0 vector field
    f :: Array{Float64,N} # force vector field
    p :: Array{Float64,M} # pressure scalar field
    σ :: Array{Float64,M} # divergence scalar field
    function Flow(u::Array{Float64,n},c::Array{Float64,n}) where n
        N = size(u); M = N[1:end-1]; m = length(M)
        @assert N==size(c)
        @assert N[end]==m
        p = zeros(M)
        f,σ = AU(N),AU(M)
        new{n,m}(u,c,f,p,σ)
    end
end

include("PoissonSys.jl")
@fastmath @inline ∇(I::CartesianIndex{2},u) = ∂(1,I,u)+∂(2,I,u)
@fastmath @inline ∇(I::CartesianIndex{3},u) = ∂(1,I,u)+∂(2,I,u)+∂(3,I,u)
@fastmath function project!(a::Flow{n,m},b::PoissonSys{n,m},Δt) where {n,m}
    @simd for I ∈ inside(a.σ)
        @inbounds a.σ[I] = ∇(I,a.u)/Δt
    end
    solve!(a.p,b,a.σ)
    for i ∈ 1:m; @simd for I ∈ inside(a.σ)
        @inbounds  a.u[I,i] -= Δt*a.c[I,i]*∂(i,I,a.p)
    end;end
end

include("PoissonSys.jl")
@fastmath function mom_step!(a::Flow,b::PoissonSys;Δt=0.25,ν=0.1,U=[1. 0.])
    fill!(a.f,0.)
    mom_transport!(a.f,a.u,ν=ν)
    @. a.u += Δt*a.c*a.f; BC!(a.u,U)
    project!(a,b,Δt); BC!(a.u,U)
end

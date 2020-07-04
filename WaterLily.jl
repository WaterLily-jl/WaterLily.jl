include("util.jl")

@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{Float64,d}) where d = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{Float64,n}) where {n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])*0.5
@fastmath quick(u,c,d) = median((5c+2d-u)/6,c,median(10c-9u,c,d))
@inline ϕu(a,I,f,u) = @inbounds u>0 ? u*quick(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*quick(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@fastmath @inline div(I::CartesianIndex{2},u) = ∂(1,I,u)+∂(2,I,u)
@fastmath @inline div(I::CartesianIndex{3},u) = ∂(1,I,u)+∂(2,I,u)+∂(3,I,u)
@fastmath @inline curl(i,I,u) = @inbounds ∂(i%3+1,CI(I,(i+1)%3+1),u)-∂((i+1)%3+1,CI(I,i%3+1),u)
@fastmath curl!(i,ω,u) = @simd for I ∈ inside(ω)
    @inbounds ω[I] = curl(i,I,u)
end
@fastmath @inline function ke(I::CartesianIndex{m},u) where m
    κ = 0.
    for i ∈ 1:m
        @inbounds κ += abs2(u[I,i]+u[I+δ(i,I),i])
    end
    return 0.25κ
end

function BC!(a::Array{T,4},A,f=1) where T
    for k∈1:size(a,3), j∈1:size(a,2)
        a[1,j,k,1] = a[2,j,k,1] = a[size(a,1),j,k,1] = f*A[1]
        a[1,j,k,2] = a[2,j,k,2]; a[size(a,1),j,k,2] = a[size(a,1)-1,j,k,2]
        a[1,j,k,3] = a[2,j,k,3]; a[size(a,1),j,k,3] = a[size(a,1)-1,j,k,3]
    end
    for k∈1:size(a,3), i∈1:size(a,1)
        a[i,1,k,2] = a[i,2,k,2] = a[i,size(a,2),k,2] = f*A[2]
        a[i,1,k,1] = a[i,2,k,1]; a[i,size(a,2),k,1] = a[i,size(a,2)-1,k,1]
        a[i,1,k,3] = a[i,2,k,3]; a[i,size(a,2),k,3] = a[i,size(a,2)-1,k,3]
    end
    for j∈1:size(a,2), i∈1:size(a,1)
        a[i,j,1,3] = a[i,j,2,3] = a[i,j,size(a,3),3] = f*A[3]
        a[i,j,1,1] = a[i,j,2,1]; a[i,j,size(a,3),1] = a[i,j,size(a,3)-1,1]
        a[i,j,1,2] = a[i,j,2,2]; a[i,j,size(a,3),2] = a[i,j,size(a,3)-1,2]
    end
end
function BC!(a::Array{T,3},A,f=1) where T
    for j∈1:size(a,2)
        a[1,j,1] = a[2,j,1] = a[size(a,1),j,1] = f*A[1]
        a[1,j,2] = a[2,j,2]; a[size(a,1),j,2] = a[size(a,1)-1,j,2]
    end
    for i∈1:size(a,1)
        a[i,1,2] = a[i,2,2] = a[i,size(a,2),2] = f*A[2]
        a[i,1,1] = a[i,2,1]; a[i,size(a,2),1] = a[i,size(a,2)-1,1]
    end
end

@fastmath function tracer_transport!(r,f,u;Pe=0.1)
    N = size(u)
    for b ∈ 1:N[end]; @simd for I ∈ inside_u(N)
        if I[b]==2 || I[b]==N[b]
            Φ = ϕ(b,I,f)*u[I,b]-Pe*∂(b,I,f)
        else
            Φ = ϕu(b,I,f,u[I,b])-Pe*∂(b,I,f)
        end
        @inbounds r[I] += Φ
        @inbounds r[I-δ(b,I)] -= Φ
    end;end
end

@fastmath function mom_transport!(r,u;ν=0.1)
    N = size(u)
    for a ∈ 1:N[end], b ∈ 1:N[end]; @simd for I ∈ inside_u(N)
        Iᵃ,Iᵇ = CI(I,a),CI(I,b)
        if Iᵇ[b]==2 || Iᵇ[b]==N[b]
            Φ = ϕ(b,Iᵃ,u)*ϕ(a,Iᵇ,u)-ν*∂(b,Iᵃ,u)
        else
            Φ = ϕu(b,Iᵃ,u,ϕ(a,Iᵇ,u))-ν*∂(b,Iᵃ,u)
        end
        @inbounds r[Iᵃ] += Φ
        @inbounds r[Iᵃ-δ(b,Iᵃ)] -= Φ
    end; end
end

struct Flow{N,M}
    u :: Array{Float64,N} # velocity vector field
    u⁰:: Array{Float64,N} # previous velocity
    μ₀:: Array{Float64,N} # BDIM zero-moment vector field
    f :: Array{Float64,N} # force vector field
    p :: Array{Float64,M} # pressure scalar field
    σ :: Array{Float64,M} # divergence scalar field
    U :: Vector{Float64}  # domain boundary values
    Δt:: Vector{Float64}  # time step
    ν :: Float64          # kinematic viscosity
    function Flow(u::Array{Float64,n},μ₀::Array{Float64,n},U;Δt=0.25,ν=0.) where n
        N = size(u); M = N[1:end-1]; m = length(M)
        @assert N==size(μ₀)
        @assert N[end]==m
        @assert length(U)==m
        u⁰ = copy(u)
        f,p,σ = zeros(N),zeros(M),zeros(M)
        new{n,m}(u,u⁰,μ₀,f,p,σ,U,[Δt],ν)
    end
end

include("PoissonSys.jl")
@fastmath function project!(a::Flow{n,m},b::Poisson{n,m}) where {n,m}
    @simd for I ∈ inside(a.σ)
        @inbounds a.σ[I] = div(I,a.u)/a.Δt[end]
    end
    solve!(a.p,b,a.σ)
    for i ∈ 1:m; @simd for I ∈ inside(a.σ)
        @inbounds  a.u[I,i] -= a.Δt[end]*a.μ₀[I,i]*∂(i,I,a.p)
    end;end
end
@fastmath function mom_step!(a::Flow,b::Poisson;O1=false,adaptive=true)
    # predictor u* = u⁰+Δtμ₀(∂Φ⁰-∂p*); ∇⋅(μ₀∂p*)=∇⋅(u⁰/Δt+μ₀∂Φ⁰)
    copy!(a.u⁰,a.u); fill!(a.f,0.)
    mom_transport!(a.f,a.u,ν=a.ν)
    @. a.u += a.Δt[end]*a.μ₀*a.f; BC!(a.u,a.U)
    project!(a,b); BC!(a.u,a.U)
    O1 && return
    # corrector u = ½(u⁰+u*+Δtμ₀(∂Φ*-∂p))
    fill!(a.f,0.)
    mom_transport!(a.f,a.u,ν=a.ν)
    @. a.u += a.u⁰+a.Δt[end]*a.μ₀*a.f; BC!(a.u,a.U,2)
    project!(a,b); a.u .*= 0.5; BC!(a.u,a.U)
    adaptive && push!(a.Δt,CFL(a))
end

function CFL(a::Flow{n,m}) where {n,m}
    @simd for I ∈ inside(a.σ)
        @inbounds a.σ[I] = ke(I,a.u)
    end
    min(1.,inv(maximum(a.σ)+3a.ν))
end

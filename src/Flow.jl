@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{Float64,d}) where d = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{Float64,n}) where {n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])*0.5
@fastmath quick(u,c,d) = median((5c+2d-u)/6,c,median(10c-9u,c,d))
@inline ϕu(a,I,f,u) = @inbounds u>0 ? u*quick(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*quick(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
@fastmath @inline div(I::CartesianIndex{m},u) where {m} = sum(∂(i,I,u) for i ∈ 1:m)

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
    N = size(u); r .= 0.
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
    V :: Array{Float64,N} # BDIM body velocity vector field
    f :: Array{Float64,N} # force vector field
    p :: Array{Float64,M} # pressure scalar field
    σ :: Array{Float64,M} # divergence scalar field
    U :: Vector{Float64}  # domain boundary values
    Δt:: Vector{Float64}  # time step
    nᵖ:: Vector{Int16}    # pressure solver iterations
    ν :: Float64          # kinematic viscosity
    function Flow(u::Array{Float64,n},μ₀::Array{Float64,n},U;Δt=0.25,ν=0.) where n
        N = size(u); M = N[1:end-1]; m = length(M)
        @assert N==size(μ₀)
        @assert N[end]==m
        @assert length(U)==m
        BC!(u,U); BC!(μ₀,zeros(m))
        V = zeros(N); BC!(V,U)
        u⁰ = copy(u)
        f,p,σ = zeros(N),zeros(M),zeros(M)
        new{n,m}(u,u⁰,μ₀,V,f,p,σ,U,[Δt],[0],ν)
    end
    function Flow(N::Tuple,U;Δt=0.25,ν=0.,uλ::Function=(i,x)->0.)
        Flow(apply(uλ,N...,length(N)),ones(N...,length(N)),U;Δt,ν)
    end
end

@fastmath function project!(a::Flow{n,m},b::AbstractPoisson{n,m}) where {n,m}
    @inside a.σ[I] = div(I,a.u)/a.Δt[end]
    i = solve!(a.p,b,a.σ)
    push!(a.nᵖ,i)
    for i ∈ 1:m; @simd for I ∈ inside(a.σ)
        @inbounds  a.u[I,i] -= a.Δt[end]*a.μ₀[I,i]*∂(i,I,a.p)
    end;end
end
@fastmath function mom_step!(a::Flow,b::AbstractPoisson)
    a.u⁰ .= a.u
    # predictor
    mom_transport!(a.f,a.u⁰,ν=a.ν)       # convection+diffusion on u⁰
    @. a.f = a.u⁰+a.Δt[end]*a.f-a.V      # 1st order BDIM
    # @inside a.u[I] = ∂∂n(I,a.f,a.μ₁n̂)    # 2nd order BDIM
    # @. a.u += a.V+a.μ₀*a.f; BC!(a.u,a.U) # momentum predictor
    @. a.u = a.V+a.μ₀*a.f; BC!(a.u,a.U)  # momentum predictor (temp)
    project!(a,b); BC!(a.u,a.U)          # pressure projection
    # corrector
    mom_transport!(a.f,a.u,ν=a.ν)        # convection+diffusion on u'
    @. a.f = a.u⁰+a.Δt[end]*a.f-a.V      # 1st order BDIM
    # @inside a.u[I] += ∂∂n(I,a.f,a.μ₁n̂)   # 2nd order BDIM + u'
    @. a.u += a.V+a.μ₀*a.f; BC!(a.u,a.U,2) # momentum corrector
    project!(a,b); a.u ./= 2; BC!(a.u,a.U) # pressure projection
    push!(a.Δt,CFL(a))
end

function CFL(a::Flow{n,m}) where {n,m}
    mx = mapreduce(max,inside(a.p)) do I
        sum(@inbounds max(0.,a.u[I,i])+max(0.,a.u[I+δ(i,I),i]) for i in 1:m)
    end
    min(10.,inv(mx+5a.ν))
end

@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{Float64,d}) where d = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{Float64,n}) where {n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@inline ∂ₐ(a,I::CartesianIndex{d},f::AbstractArray{Float64,d}) where d = @inbounds 0.5*(f[I+δ(a,I)]-f[I-δ(a,I)])
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])*0.5
@fastmath quick(u,c,d) = median((5c+2d-u)/6,c,median(10c-9u,c,d))
@fastmath vanLeer(u,c,d) = (c≤min(u,d) || c≥max(u,d)) ? c : c+(d-c)*(c-u)/(d-u)
@inline ϕu(a,I,f,u,λ=quick) = @inbounds u>0 ? u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])
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

@fastmath function conv_diff!(r::Array{T,m},u::Array{T,m};ν=0.1) where {T,m}
    r .= 0.
    n = m-1; N = ntuple(i -> size(u,i), n)
    for i ∈ 1:n, j ∈ 1:n
        @simd for I ∈ slice(N,2,j,2)
            Φ = ϕ(j,CI(I,i),u)*ϕ(i,CI(I,j),u)-ν*∂(j,CI(I,i),u)
            @inbounds r[I,i] += Φ
        end
        @simd for I ∈ inside_u(N,j)
            Φ = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u))-ν*∂(j,CI(I,i),u)
            @inbounds r[I,i] += Φ
            @inbounds r[I-δ(j,I),i] -= Φ
        end
        @simd for I ∈ slice(N,N[j],j,2)
            Φ = ϕ(j,CI(I,i),u)*ϕ(i,CI(I,j),u)-ν*∂(j,CI(I,i),u)
            @inbounds r[I-δ(j,I),i] -= Φ
        end
    end
end

struct Flow{N,M,P}
    # Fluid fields
    u :: Array{Float64,M} # velocity vector
    u⁰:: Array{Float64,M} # previous velocity
    f :: Array{Float64,M} # force vector
    p :: Array{Float64,N} # pressure scalar
    σ :: Array{Float64,N} # divergence scalar
    # BDIM fields
    V :: Array{Float64,M} # body velocity vector
    σᵥ:: Array{Float64,N} # body velocity divergence
    μ₀:: Array{Float64,M} # zeroth-moment on faces
    μ₁:: Array{Float64,P} # first-moment vector on faces
    # Non-fields
    U :: Vector{Float64}  # domain boundary values
    Δt:: Vector{Float64}  # time step
    ν :: Float64          # kinematic viscosity
    function Flow(N::Tuple,U::Vector;Δt=0.25,ν=0.,uλ::Function=(i,x)->0.)
        d = length(N); Nd = (N...,d)
        @assert length(U)==d
        u = apply(uλ,Nd); BC!(u,U)
        u⁰ = copy(u)
        f,p,σ = zeros(Nd),zeros(N),zeros(N)
        V,σᵥ = zeros(Nd),zeros(N)
        μ₀ = ones(Nd); BC!(μ₀,zeros(d))
        μ₁ = zeros(N...,d,d)
        new{d,d+1,d+2}(u,u⁰,f,p,σ,V,σᵥ,μ₀,μ₁,U,[Δt],ν)
    end
end

@fastmath function BDIM!(a::Flow{n}) where n
    @. a.f = a.u⁰+a.Δt[end]*a.f-a.V
    for j ∈ 1:n, i ∈ 1:n; @simd for I ∈ inside(a.p)
        @inbounds a.u[I,i] += a.μ₁[I,i,j]*∂ₐ(j,CI(I,i),a.f)
    end;end
    @. a.u += a.V+a.μ₀*a.f
end

@fastmath function project!(a::Flow{n},b::AbstractPoisson{n},w=1) where n
    @inside a.σ[I] = (div(I,a.u)+w*a.σᵥ[I])/a.Δt[end]
    solver!(a.p,b,a.σ)
    for i ∈ 1:n; @simd for I ∈ inside(a.σ)
        @inbounds  a.u[I,i] -= a.Δt[end]*a.μ₀[I,i]*∂(i,I,a.p)
    end;end
end

@fastmath function mom_step!(a::Flow,b::AbstractPoisson)
    a.u⁰ .= a.u; a.u .= 0
    # predictor u → u'
    conv_diff!(a.f,a.u⁰,ν=a.ν)
    BDIM!(a); BC!(a.u,a.U)
    project!(a,b); BC!(a.u,a.U)
    # corrector u → u¹
    conv_diff!(a.f,a.u,ν=a.ν)
    BDIM!(a); BC!(a.u,a.U,2)
    project!(a,b,2); a.u ./= 2; BC!(a.u,a.U)
    push!(a.Δt,CFL(a))
end

@fastmath @inline fout(I::CartesianIndex{d},u) where {d} =
    sum(@inbounds(max(0.,u[I+δ(a,I),a])+max(0.,-u[I,a])) for a ∈ 1:d)
function CFL(a::Flow{n}) where n
    mx = mapreduce(I->fout(I,a.u),max,inside(a.p))
    min(10.,inv(mx+5a.ν))
end

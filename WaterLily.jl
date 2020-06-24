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

function BC!(u::Array{T,d},U) where {T<:Real, d}
    for a ∈ 1:d-1, b ∈ 1:d-1 # components, faces
        a==b ?
        _dirichlet(u,b,CR(size(u)[1:b-1]),CR(size(u)[b+1:end-1]),a,U[a]) :
        _nuemann(u,b,CR(size(u)[1:b-1]),CR(size(u)[b+1:end-1]),a)
    end
end
function BC!(f::Array{T,d}) where {T<:Real, d}
    for b ∈ 1:d # domain faces
        _nuemann(f,b,CR(size(f)[1:b-1]),CR(size(f)[b+1:end]),CI())
    end
end
function _nuemann(f,b,left,right,a)
    for r ∈ right, l ∈ left
        f[l,1,r,a] = f[l,2,r,a]; f[l,size(f,b),r,a] = f[l,size(f,b)-1,r,a]
    end
end
function _dirichlet(f,b,left,right,a,F)
    for r ∈ right, l ∈ left
        f[l,1,r,a] = f[l,2,r,a] = f[l,size(f,b),r,a] = F
    end
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

include("AMGproject.jl")
struct flow
    u;c;r
    p
    aml;σ;p_vec
end
function flow(u,c)
    n,m,d = size(u)
    flow(u,c,similar(u),zeros(n,m),AMG(c),zeros((n-2)*(m-2)),zeros((n-2)*(m-2)))
end

@fastmath function mom_step!(a::flow;Δt=0.25,ν=0.1,U=[1. 0.])
    fill!(a.r,0.)
    mom_transport!(a.r,a.u,ν=ν)
    @. a.u += Δt*a.c*a.r; BC!(a.u,U)
    projectAMG!(a.p,a.u,a.c,a.σ,a.p_vec,a.aml,Δt)
    BC!(a.u,U);
end

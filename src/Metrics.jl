using StaticArrays

# utilities
Base.@propagate_inbounds @inline fSV(f,n) = SA[ntuple(f,n)...]
Base.@propagate_inbounds @inline @fastmath fsum(f,n) = sum(ntuple(f,n))
norm2(x) = √(x'*x)
Base.@propagate_inbounds @fastmath function permute(f,i)
    j,k = i%3+1,(i+1)%3+1
    f(j,k)-f(k,j)
end
×(a,b) = fSV(i->permute((j,k)->a[j]*b[k],i),3)

"""
    ke(I::CartesianIndex,u,U=0)

Compute ``½∥𝐮-𝐔∥²`` at center of cell `I` where `U` can be used
to subtract a background flow (by default, `U=0`).
"""
ke(I::CartesianIndex{m},u,U=fSV(zero,m)) where m = 0.125fsum(m) do i
    abs2(@inbounds(u[I,i]+u[I+δ(i,I),i]-2U[i]))
end
"""
    ∂(i,j,I,u)

Compute ``∂uᵢ/∂xⱼ`` at center of cell `I`. Cross terms are computed
less accurately than inline terms because of the staggered grid.
"""
@fastmath @inline ∂(i,j,I,u) = (i==j ? ∂(i,I,u) :
        @inbounds(u[I+δ(j,I),i]+u[I+δ(j,I)+δ(i,I),i]
                 -u[I-δ(j,I),i]-u[I-δ(j,I)+δ(i,I),i])/4)

using LinearAlgebra: eigvals
"""
    λ₂(I::CartesianIndex{3},u)

λ₂ is a deformation tensor metric to identify vortex cores.
See [https://en.wikipedia.org/wiki/Lambda2_method](https://en.wikipedia.org/wiki/Lambda2_method) and
Jeong, J., & Hussain, F., doi:[10.1017/S0022112095000462](https://doi.org/10.1017/S0022112095000462)
"""
function λ₂(I::CartesianIndex{3},u)
    J = @SMatrix [∂(i,j,I,u) for i ∈ 1:3, j ∈ 1:3]
    S,Ω = (J+J')/2,(J-J')/2
    eigvals(S^2+Ω^2)[2]
end

"""
    curl(i,I,u)

Compute component `i` of ``𝛁×𝐮`` at the __edge__ of cell `I`.
For example `curl(3,CartesianIndex(2,2,2),u)` will compute
`ω₃(x=1.5,y=1.5,z=2)` as this edge produces the highest
accuracy for this mix of cross derivatives on a staggered grid.
"""
curl(i,I,u) = permute((j,k)->∂(j,CI(I,k),u), i)
"""
    ω(I::CartesianIndex{3},u)

Compute 3-vector ``𝛚=𝛁×𝐮`` at the center of cell `I`.
"""
ω(I::CartesianIndex{3},u) = fSV(i->permute((j,k)->∂(k,j,I,u),i),3)
"""
    ω_mag(I::CartesianIndex{3},u)

Compute ``∥𝛚∥`` at the center of cell `I`.
"""
ω_mag(I::CartesianIndex{3},u) = norm2(ω(I,u))
"""
    ω_θ(I::CartesianIndex{3},z,center,u)

Compute ``𝛚⋅𝛉`` at the center of cell `I` where ``𝛉`` is the azimuth
direction around vector `z` passing through `center`.
"""
function ω_θ(I::CartesianIndex{3},z,center,u)
    θ = z × (loc(0,I)-SVector{3}(center))
    n = norm2(θ)
    n<=eps(n) ? 0. : θ'*ω(I,u) / n
end
"""
    ∮nds(p,body::AutoBody,t=0)

Surface normal integral of field `p` over the `body`.
"""
∮nds(flow::Flow,body::AbstractBody) = ∮nds(flow.p,flow.f,body,time(flow))
function ∮nds(p::AbstractArray{T,N},df::AbstractArray{T},body::AbstractBody,t=0) where {T,N}
    @fastmath @inline function fill!(df,p,I)
        d = sdf(body,WaterLily.loc(0,I,T),t)
        if abs(d)<1
            dᵢ,nᵢ,_ = measure(body,WaterLily.loc(0,I,T),t)
            df[I,:] .= p[I]*nᵢ*WaterLily.kern(clamp(dᵢ,-1,1))
        else
            df[I,:] .= 0
        end
    end
    @loop fill!(df,p,I) over I ∈ inside(p)
    T[sum(@inbounds(df[inside(p),i])) for i ∈ 1:N] |> Array
end
# viscous stress tensor
∇²u(I::CartesianIndex{2},u) = @SMatrix [∂(i,j,I,u)+∂(j,i,I,u) for i ∈ 1:2, j ∈ 1:2]
∇²u(I::CartesianIndex{3},u) = @SMatrix [∂(i,j,I,u)+∂(j,i,I,u) for i ∈ 1:3, j ∈ 1:3]
"""
   ∮τnds(u::AbstractArray{T,N},df::AbstractArray{T},body::AbstractBody,t=0)

Compute the viscous force on a immersed body. 
"""
∮τnds(flow::Flow,body::AbstractBody) = ∮τnds(flow.u,flow.f,body,time(flow))
function ∮τnds(u::AbstractArray{T,N},df::AbstractArray{T,N},body::AbstractBody,t=0) where {T,N}
    @warn "∮τnds is can be innacurate for viscous force computation, use with care..."
    Nu,_ = size_u(u); In = CartesianIndices(map(i->(2:i-1),Nu))
    @fastmath @inline function fill!(df,u,I)
        d = sdf(body,WaterLily.loc(0,I,T),t)
        if abs(d)<1
            dᵢ,nᵢ,_ = measure(body,WaterLily.loc(0,I,T),t)
            df[I,:] .= ∇²u(I,u)*nᵢ*WaterLily.kern(clamp(dᵢ,-1,1))
        else
            df[I,:] .= 0
        end
    end
    @loop fill!(df,u,I) over I ∈ inside(In)
    T[sum(@inbounds(df[inside(In),i])) for i ∈ 1:N-1] |> Array
end
using LinearAlgebra: cross
"""
    ∮xnds(x₀::Svector,u::AbstractArray{T,N},df::AbstractArray{T},body::AbstractBody,t=0)

Computes the pressure moment on a immersed body relative to point x₀. 
"""
∮xnds(x₀,flow::Flow,body::AbstractBody) = ∮xnds(x₀,flow.p,flow.f,body,time(flow))
function ∮xnds(x₀::SVector{N,T},p::AbstractArray{T,N},df::AbstractArray{T},body::AbstractBody,t=0) where {N,T}
    @fastmath @inline function fill!(df,p,I)
        d = sdf(body,WaterLily.loc(0,I,T),t)
        if abs(d)<1
            dᵢ,nᵢ,_ = measure(body,WaterLily.loc(0,I,T),t)
            df[I,:] .= p[I]*cross((WaterLily.loc(0,I,T)-x₀),nᵢ*WaterLily.kern(clamp(dᵢ,-1,1)))
        else
            df[I,:] .= 0
        end
    end
    @loop fill!(df,p,I) over I ∈ inside(p)
    T[sum(@inbounds(df[inside(p),i])) for i ∈ 1:N] |> Array
end
∮xnds(x₀,flow::Flow{2},body::AbstractBody) = ∮xnds(x₀,flow.p,flow.σ,body,time(flow))
function ∮xnds(x₀::SVector{2,T},p::AbstractArray{T,2},σ::AbstractArray{T,2},body::AbstractBody,t=0) where T
    @fastmath @inline function fill!(df,p,I)
        d = sdf(body,WaterLily.loc(0,I,T),t)
        if abs(d)<1
            dᵢ,nᵢ,_ = measure(body,WaterLily.loc(0,I,T),t)
            df[I] = p[I]*cross((WaterLily.loc(0,I,T)-x₀),nᵢ*WaterLily.kern(clamp(dᵢ,-1,1)))
        else
            df[I] = 0
        end
    end
    @loop fill!(σ,p,I) over I ∈ inside(p)
    sum(@inbounds(σ[inside(p)]))
end
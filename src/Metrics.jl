using StaticArrays

# utilities
@inline fSV(f,n) = SA[ntuple(f,n)...]
@inline @fastmath fsum(f,n) = sum(ntuple(f,n))
norm2(x) = √(x'*x)
@fastmath function permute(f,i)
    j,k = i%3+1,(i+1)%3+1
    f(j,k)-f(k,j)
end
×(a,b) = fSV(i->permute((j,k)->a[j]*b[k],i),3)

"""
    ke(I::CartesianIndex,u,U=0)

Compute ½|u-U|² at center of cell `I` where `U` can be used
to subtract a background flow.
"""
ke(I::CartesianIndex{m},u,U=fSV(zero,m)) where m = 0.125fsum(m) do i
    abs2(@inbounds(u[I,i]+u[I+δ(i,I),i]-2U[i]))
end
"""
    ∂(i,j,I,u)

Compute ∂uᵢ/∂xⱼ at center of cell `I`. Cross terms are computed
less accurately than inline terms because of the staggered grid.
"""
@fastmath @inline ∂(i,j,I,u) = (i==j ? ∂(i,I,u) :
        @inbounds(u[I+δ(j,I),i]+u[I+δ(j,I)+δ(i,I),i]
                 -u[I-δ(j,I),i]-u[I-δ(j,I)+δ(i,I),i])/4)

using LinearAlgebra: eigvals
"""
    λ₂(I::CartesianIndex{3},u)

λ₂ is a deformation tensor metric to identify vortex cores.
See https://en.wikipedia.org/wiki/Lambda2_method and
Jeong, J., & Hussain, F. doi:10.1017/S0022112095000462
"""
function λ₂(I::CartesianIndex{3},u)
    J = @SMatrix [∂(i,j,I,u) for i ∈ 1:3, j ∈ 1:3]
    S,Ω = (J+J')/2,(J-J')/2
    eigvals(S^2+Ω^2)[2]
end

"""
    curl(i,I,u)

Compute component `i` of ∇×u at the __edge__ of cell `I`.
For example `curl(3,CartesianIndex(2,2,2),u)` will compute
`ω₃(x=1.5,y=1.5,z=2)` as this edge produces the highest
accuracy for this mix of cross derivatives on a staggered grid.
"""
curl(i,I,u) = permute((j,k)->∂(j,CI(I,k),u), i)
"""
    ω(I::CartesianIndex{3},u)

Compute 3-vector ω=∇×u at the center of cell `I`.
"""
ω(I::CartesianIndex{3},u) = fSV(i->permute((j,k)->∂(k,j,I,u),i),3)
"""
    ω_mag(I::CartesianIndex{3},u)

Compute |ω| at the center of cell `I`.
"""
ω_mag(I::CartesianIndex{3},u) = norm2(ω(I,u))
"""
    ω_θ(I::CartesianIndex{3},z,center,u)

Compute ω⋅θ at the center of cell `I` where θ is the azimuth
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
function ∮nds(p::AbstractArray{T,N},df::AbstractArray{T},body::AutoBody,t=0) where {T,N}
    nds!(df,body,t)
    for i in 1:N
        @loop df[I,i] = df[I,i]*p[I] over I ∈ inside(p)
    end
    reshape(sum(df,dims=1:N),N) |> Array
end
nds!(a,body,t=0) = apply!(a) do i,x
    d = body.sdf(x,t)
    n = ForwardDiff.gradient(y -> body.sdf(y,t), x)
    n[i]*WaterLily.kern(clamp(d,-1,1))
end

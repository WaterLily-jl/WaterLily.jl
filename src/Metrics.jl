using LinearAlgebra: eigvals,norm2,×,⋅
"""
    ke(I::CartesianIndex,u,U=0)

Compute ½|u-U|² at center of cell `I` where `U` can be used
to subtract a background flow.
"""
ke(I::CartesianIndex{m},u,U=zeros(m)) where m = 0.125sum(@inbounds(abs2(u[I,i]+u[I+δ(i,I),i]-2U[i])) for i ∈ 1:m)
"""
    ∂(i,j,I,u)

Compute ∂uᵢ/∂xⱼ at center of cell `I`. Cross terms are computed
less accurately than inline terms because of the staggered grid.
"""
@fastmath @inline ∂(i,j,I,u) = (i==j ? ∂(i,I,u) :
    @inbounds 0.25*(u[I+δ(j,I),i]+u[I+δ(j,I)+δ(i,I),i]
                   -u[I-δ(j,I),i]-u[I-δ(j,I)+δ(i,I),i]))
"""
    λ₂(I::CartesianIndex{3},u)

λ₂ is a deformation tensor metric to identify vortex cores.
See https://en.wikipedia.org/wiki/Lambda2_method and
Jeong, J., & Hussain, F. doi:10.1017/S0022112095000462
"""
function λ₂(I::CartesianIndex{3},u)
    J = [∂(i,j,I,u) for i ∈ 1:3, j ∈ 1:3]
    S,Ω = (J+J')/2,(J-J')/2
    eigvals(S^2+Ω^2)[2]
end

function permute(f,i)
    j,k = i%3+1,(i+1)%3+1
    f(j,k)-f(k,j)
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
ω(I::CartesianIndex{3},u) = [permute((j,k)->∂(k,j,I,u), i) for i ∈ 1:3]
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
    θ = z × (loc(0,I)-center)
    n = norm2(θ)
    n<=eps(n) ? 0. : θ ⋅ ω(I,u) / n
end
"""
    ∮nds(p,body::AutoBody,t=0)

Surface normal integral of field `p` over the `body`.
"""
function ∮nds(p::Array{T,N},body::AutoBody,t=0) where {T,N}
    s = zeros(SVector{N,T})
    n = x -> ForwardDiff.gradient(y -> body.sdf(y,t), x)
    for I ∈ inside(p)
        x = loc(0,I)
        d = body.sdf(x,t)::Float64
        abs(d) ≤ 1 && (s += n(x).*p[I]*WaterLily.kern(d))
    end
    return s
end

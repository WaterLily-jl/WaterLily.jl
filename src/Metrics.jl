using LinearAlgebra: eigvals
using StaticArrays

@fastmath @inline ∂(i,j,I,u) = (i==j ? ∂(i,I,u) :
    @inbounds 0.25*(u[I+δ(j,I),i]+u[I+δ(j,I)+δ(i,I),i]
                   -u[I-δ(j,I),i]-u[I-δ(j,I)+δ(i,I),i]))
@fastmath @inline curl(i,I,u) = @inbounds ∂(i%3+1,CI(I,(i+1)%3+1),u)-∂((i+1)%3+1,CI(I,i%3+1),u)
@fastmath @inline ke(I::CartesianIndex{m},u) where m = 0.125sum(@inbounds(abs2(u[I,i]+u[I+δ(i,I),i])) for i ∈ 1:m)
"""
λ₂ is a deformation tensor metric to identify vortex cores
See https://en.wikipedia.org/wiki/Lambda2_method and
Jeong, J., & Hussain, F. doi:10.1017/S0022112095000462
"""
function λ₂(I::CartesianIndex{3},u)
    J = @SMatrix [∂(i,j,I,u) for i ∈ 1:3, j ∈ 1:3]
    S,Ω = (J+J')/2,(J-J')/2
    eigvals(S^2+Ω^2)[2]
end

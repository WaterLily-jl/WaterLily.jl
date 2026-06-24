using EllipsisNotation
"""
    interp(x::SVector, arr::AbstractArray)

Linear interpolation from array `arr` at Cartesian-coordinate `x`. Interpolation queries are clamped to the computational domain.
Note: This routine works for any number of dimensions.

To interpolate from an `arr<:GPUArray`, the call for `interp` should be broadcasted over the coordinates `x` as follows:
```julia
p = CUDA.rand(10,18)
u = CUDA.rand(10,18,2)
x = CuArray([SA_F32[i-1.5, 2i+0.5] for i in 1:8])
WaterLily.interp.(x, Ref(p)) # Broadcast
WaterLily.interp.(x, Ref(u)) # Broadcast (x=[-0.5,2.5] is shifted to [0,2.5] because we are in a vector field)
```
"""
@inline _interp_clamp(x::SVector{D,T}, sz::NTuple{D,Int}) where {D,T} =
    SVector{D,T}(clamp(x[d], zero(T), T(sz[d] - 2)) for d in 1:D)

function interp(x::SVector{D,T}, varr::AbstractArray{T}) where {D,T}
    # Each component is stored on a staggered face, so shift query for that
    # component and then clamp to the valid scalar interpolation domain.
    @inline shift(i) = SVector{D,T}(ifelse(i==j,0.5,0.) for j in 1:D)
    return SVector{D,T}(_interp(_interp_clamp(x + shift(i), size(varr)[1:D]), @view(varr[..,i])) for i in 1:D)
end
function interp(x::SVector{D,T}, arr::AbstractArray{T,D}) where {D,T}
    _interp(_interp_clamp(x, size(arr)), arr)
end
function _interp(x::SVector{D,T}, arr::AbstractArray{T,D}) where {D,T}
    # Index below the interpolation coordinate and the difference
    x = x .+ 1.5f0; i = floor.(Int,x); y = x.-i

    # CartesianIndices around x
    I = CartesianIndex(i...); R = I:I+oneunit(I)

    # Linearly weighted sum over arr[R] (in serial)
    s = zero(T)
    @fastmath @inbounds @simd for J in R
        weight = prod(@. ifelse(J.I==I.I,1-y,y))
        s += arr[J]*weight
    end
    return s
end

"""
    sgs!(flow, u, t; νₜ, S, Cs, Δ)

Implements a user-defined function `udf` to model subgrid-scale LES stresses based on the Boussinesq approximation
    τᵃᵢⱼ = τʳᵢⱼ - (1/3)τʳₖₖδᵢⱼ = -2νₜS̅ᵢⱼ
where
            ▁▁▁▁
    τʳᵢⱼ =  uᵢuⱼ - u̅ᵢu̅ⱼ

and we add -∂ⱼ(τᵃᵢⱼ) to the RHS as a body force (the isotropic part of the tensor is automatically modelled by the pressure gradient term).
Users need to define the turbulent viscosity function `νₜ` and pass it as a keyword argument to this function together with rate-of-strain
tensor array buffer `S`, Smagorinsky constant `Cs`, and filter width `Δ`.
For example, the standard Smagorinsky–Lilly model for the sub-grid scale stresses is

    νₜ = (CₛΔ)²|S̅ᵢⱼ|=(CₛΔ)²√(2S̅ᵢⱼS̅ᵢⱼ)

It can be implemented as
    `smagorinsky(I::CartesianIndex{m} where m; S, Cs, Δ) = @views (Cs*Δ)^2*sqrt(dot(S[I,:,:],S[I,:,:]))`
and passed into `sim_step!` as a keyword argument together with the varibles than the function needs (`S`, `Cs`, and `Δ`):
    `sim_step!(sim, ...; udf=sgs, νₜ=smagorinsky, S, Cs, Δ)`
"""
function sgs!(flow, u, t; νₜ, S, Cs, Δ)
    N,n = size_u(u)
    @loop S[I,:,:] .= WaterLily.S(I,u) over I ∈ inside(flow.σ)
    for i ∈ 1:n, j ∈ 1:n
        WaterLily.@loop (
            flow.σ[I] = -νₜ(I;S,Cs,Δ)*∂(j,CI(I,i),u);
            flow.f[I,i] += flow.σ[I];
        ) over I ∈ inside_u(N,j)
        WaterLily.@loop flow.f[I-δ(j,I),i] -= flow.σ[I] over I ∈ WaterLily.inside_u(N,j)
    end
end

squeeze(a::AbstractArray) = dropdims(a, dims = tuple(findall(size(a) .== 1)...))

"""
    spread!(sim3D, sim2D; dim=3, ϵ=0)

Spread a given 2D `Simulation` onto a 3D `Simulation` by extruding it along the dim `dim`.

Default is to extrude along the `dim=3`, user can also pass in a given noise level `ϵ` that is
applied to perturb the velocity field. The pressure field is left unchanged.
Internally, the function test that that the 3D `Simulation` is exactly an extruded version of
the 2D Simulation, i.e. the body must match through μ₀.

Example:
```julia
# 2D or 3D cylinder
body = AutoBody((x,t)->√sum(abs2,SA[x[1]-8,x[2]-8])-6)
# the sims
sim2D = Simulation((32,16)  ,(1.0,0.0)    ,1.0;body)
sim3D = Simulation((32,16,8),(1.0,0.0,0.0),1.0;body,perdir=(3,))
# spread after a few steps
sim_step!(sim2D,100)
WaterLily.spread!(sim3D, sim2D; dim=3, ϵ=0.0)
```
"""
function spread!(sim3D::AbstractSimulation, sim2D::AbstractSimulation; dim=3, ϵ=0)
    T,S = eltype(sim2D.flow.p), size(sim3D.flow.p)
    size3D = ntuple(j->j<dim ? S[j] : S[j+1], 2)
    @assert size(sim2D.flow.p)==size3D "Spread dimensions mismatch between sim2D $(size(sim2D.flow.p)) and sim3D $(size3D) for dim $(dim)"
    Is = CartesianIndices(((ntuple(j->j==dim ? (1:1) : (1:S[j]), 3))..., 1:2))
    @assert all(sim2D.flow.μ₀ .≈ squeeze(sim3D.flow.μ₀[Is])) "There seem to be a body mistmatch between the body in the sim2D and the sim3D along dim $(dim)"
    spread!(sim3D.flow.p, sim2D.flow.p; dim=dim, ϵ=zero(T))
    spread!(sim3D.flow.u, sim2D.flow.u; dim=dim, ϵ=T(ϵ))
end

"""
    spread!(src:AbstractArray{T,N}, dest::AbstractArray{T,N+1}; ϵ=0, dims=3)

Spreads a `N` dim field into a `N+1` field. The parameter `ϵ` sets the random noise added to the spread and
`dims` specifies the dimension along which the spreading is done.

```julia
dest = zeros(20,10,5)
src  = rand(20,10)
WaterLily.spread!(src, dest; ϵ=0.01, dims=3)
```
"""
spread!(dest::AbstractArray{T,3}, src::AbstractArray{T,2}; dim=3, ϵ=zero(T)) where T = (@loop dest[I] = src[dropindex(I,dim)]+ϵ*rand() over I in CartesianIndices(dest))
spread!(dest::AbstractArray{T,4}, src::AbstractArray{T,3}; dim=3, ϵ=zero(T)) where T = for i in 1:2
    @loop dest[I,i] = src[dropindex(I,dim),i]+ϵ*rand() over I in CartesianIndices(size(dest)[1:3])
end
@inline dropindex(I::CartesianIndex{N}, i::Int) where N = CartesianIndex(ntuple(j -> j<i ? I.I[j] : I.I[j+1], Val(N-1)))

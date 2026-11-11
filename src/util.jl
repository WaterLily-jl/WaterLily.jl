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
    # Component i is stored on face i (see `stagger`), so shift the query to it and clamp
    return SVector{D,T}(_interp(_interp_clamp(x .- stagger(i,D,T), size(varr)[1:D]), varr, i) for i in 1:D)
end
interp(x::SVector{D,T}, arr::AbstractArray{T,D}) where {D,T} = _interp(_interp_clamp(x, size(arr)), arr)
function _interp(x::SVector{D,T}, arr::AbstractArray{T}, c...) where {D,T}
    x = x .+ 1.5f0; I = floor.(Int,x); y = x.-I
    sum(ntuple(Val(1<<D)) do k
        o = ntuple(d -> ((k-1)>>(d-1)) & 1, D)
        @inbounds arr[CartesianIndex(Tuple(I) .+ o), c...] * prod(ntuple(d -> o[d]==1 ? y[d] : 1-y[d], D))
    end)
end

"""
    sgs!(flow, u, t; νₜ, S=nothing, ∇u=nothing, kwargs...)

Implements a user-defined function `udf` to model subgrid-scale LES stresses based on the Boussinesq approximation
    τᵃᵢⱼ = τʳᵢⱼ - (1/3)τʳₖₖδᵢⱼ = -2νₜS̅ᵢⱼ
where
            ▁▁▁▁
    τʳᵢⱼ =  uᵢuⱼ - u̅ᵢu̅ⱼ

and we add -∂ⱼ(τᵃᵢⱼ) = ∂ⱼ(νₜ(∂ⱼuᵢ+∂ᵢuⱼ)) to the RHS as a body force (the isotropic part of the tensor is automatically
modelled by the pressure gradient term). The fluxes are evaluated where `conv_diff!` evaluates its own, with `νₜ` taken at
the cell centre for the normal stresses and averaged over the four cells around an edge for the shear stresses, on all
the faces of the domain, periodic or not.

Users need to define the turbulent viscosity function `νₜ(I::CartesianIndex; kwargs...)`, evaluated at the centre of cell `I`,
and pass it as a keyword argument to this function together with the buffers and parameters it needs:
  - `S`: array of size `(size(flow.p)...,D,D)`, filled with the rate-of-strain tensor `S̅ᵢⱼ` and passed on to `νₜ`.
  - `∇u`: array of the same size, filled with the velocity gradient `∂uᵢ/∂xⱼ` and passed on to `νₜ`.
  - any other keyword argument is passed on to `νₜ` as it is.

For example, the standard Smagorinsky–Lilly model for the sub-grid scale stresses is

    νₜ = (CₛΔ)²|S̅ᵢⱼ|=(CₛΔ)²√(2S̅ᵢⱼS̅ᵢⱼ)

It can be implemented with an explicit loop, which compiles inside the kernels on every backend, as

    @inline function smagorinsky(I::CartesianIndex{D}; S, Cs, Δ) where D
        s = zero(eltype(S))
        for i in 1:D, j in 1:D
            s += S[I,i,j]^2
        end
        (Cs*Δ)^2*√(2s)
    end

and passed into `sim_step!` as a keyword argument together with the variables that the function needs (`S`, `Cs`, and `Δ`):

    sim_step!(sim, ...; udf=WaterLily.sgs!, νₜ=smagorinsky, S, Cs, Δ)

Models built on the full velocity gradient, which vanish in a pure shear (Vreman, WALE, σ), take the buffer `∇u` instead.
`flow.σ` holds `νₜ` on return.
"""
function sgs!(flow, u, t; νₜ, S=nothing, ∇u=nothing, kwargs...)
    isnothing(S) || @loop S[I,:,:] .= WaterLily.S(I,u) over I ∈ inside(flow.σ)
    isnothing(∇u) || @loop ∇u[I,:,:] .= WaterLily.∇u(I,u) over I ∈ inside(flow.σ)
    kw = (;(isnothing(S) ? (;) : (;S))...,(isnothing(∇u) ? (;) : (;∇u))...,kwargs...)
    @loop flow.σ[I] = νₜ(I;kw...) over I ∈ inside(flow.σ)
    neumannBC!(flow.σ,flow.perdir)
    @loop flow.f[Ii] += sgs_force(Ii,u,flow.σ) over Ii ∈ inside_u(size(flow.σ))
end

# Zero-gradient ghost cells of a scalar field, periodic along `perdir`
function neumannBC!(a,perdir=())
    N = size(a)
    for j ∈ eachindex(N)
        j in perdir && continue
        @loop a[I] = a[I+δ(j,I)] over I ∈ slice(N,1,j)
        @loop a[I] = a[I-δ(j,I)] over I ∈ slice(N,N[j],j)
    end
    perBC!(a,perdir)
end

"""
    sgs_flux(i,j,I,u,νₜ)

SGS flux `νₜ(∂ⱼuᵢ+∂ᵢuⱼ)` of the momentum component `i` through the lower face along `j` of the control volume of `u[I,i]`,
where `conv_diff!` evaluates its flux `Φ`. This is the centre of cell `I-δ(i,I)` when `i==j` and an edge of cell `I` otherwise,
where the eddy viscosity `νₜ`, given at the cell centres, is averaged over the four cells around it.
"""
@fastmath @inline function sgs_flux(i,j,I::CartesianIndex,u,νₜ)
    ν = @inbounds i==j ? νₜ[I-δ(i,I)] : (νₜ[I]+νₜ[I-δ(i,I)]+νₜ[I-δ(j,I)]+νₜ[I-δ(i,I)-δ(j,I)])/4
    ν*(∂(j,CI(I,i),u)+∂(i,CI(I,j),u))
end
# SGS force on the face `Ii`: divergence of `sgs_flux` over its control volume
@fastmath @inline function sgs_force(Ii::CartesianIndex{np1},u,νₜ) where np1
    I,i = CartesianIndex(Base.front(Ii.I)),last(Ii.I); s = zero(eltype(u))
    for j ∈ 1:np1-1
        s += sgs_flux(i,j,I+δ(j,I),u,νₜ)-sgs_flux(i,j,I,u,νₜ)
    end
    return s
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
    spread!(src:AbstractArray{T,N}, dest::AbstractArray{T,N+1}; ϵ=0, dim=3)

Spreads a `N` dim field into a `N+1` field. The parameter `ϵ` sets the random noise added to the spread and
`dim` specifies the dimension along which the spreading is done.

```julia
dest = zeros(20,10,5)
src  = rand(20,10)
WaterLily.spread!(src, dest; ϵ=0.01, dim=3)
```
"""
spread!(dest::AbstractArray{T,3}, src::AbstractArray{T,2}; dim=3, ϵ=zero(T)) where T = (@loop dest[I] = src[dropindex(I,dim)]+ϵ*rand() over I in CartesianIndices(dest))
spread!(dest::AbstractArray{T,4}, src::AbstractArray{T,3}; dim=3, ϵ=zero(T)) where T = for i in 1:2
    @loop dest[I,i] = src[dropindex(I,dim),i]+ϵ*rand() over I in CartesianIndices(size(dest)[1:3])
end
@inline dropindex(I::CartesianIndex{N}, i::Int) where N = CartesianIndex(ntuple(j -> j<i ? I.I[j] : I.I[j+1], Val(N-1)))

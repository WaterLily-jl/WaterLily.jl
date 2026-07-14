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

# ----------------------------------------------------------------------------
# Implicit-LES parametrized dissipative numerical flux (energy-conserving cds
# base + JST/Rusanov graded artificial dissipation), tuned by `β` via AD.
#
# For each momentum component `i` and transport direction `j`, the dissipative
# face flux at the j-face `I` (between cells `I-δⱼ` and `I`) is
#     fᵈ = -½ |U_face| Σₚ βₚ Δ^{2p}ⱼ uᵢ ,   U_face = ϕ(i,CI(I,j),u)  (= the
# transporting velocity u_j interpolated to the i-face, exactly as in `ϕu`).
# It is added with the SAME telescoping as the convective flux (`f[I,i]+=fᵈ`,
# `f[I-δⱼ,i]-=fᵈ`), so it is momentum-conservative, and on periodic directions the
# wrap-face is closed exactly as in `conv_diff!` (no seam dissipation deficit).
# SIGN: with the all-plus Pascal stencil (as in the Burgers reference) the per-order
# dissipative sign is sign(βₚ)=(-1)^(p+1): β₁,β₃>0 REMOVE energy (Δ²,Δ⁶), but β₂,β₄
# remove energy only when NEGATIVE (Δ⁴,Δ⁸ INJECT energy for βₚ>0). So box-constrain
# β₁,β₃≥0 and β₂,β₄≤0 (or leave even orders free). At P=1, β₁>0 is Rusanov/
# Smagorinsky-like (Cs²↔β₁/2). Use with the energy-conserving `cds` base — set
# `λ=cds` at `Simulation`/`Flow` construction (the scheme lives in `flow.λ`; a `λ`
# kwarg to `sim_step!` is silently ignored) — and `udf=dissipative_flux!`; do NOT
# combine with quick/vanLeer or `sgs!` (double-counts).
#
# Δ^{2p} are the central undivided even differences of Burgers' d2/d4/d6/d8
# (Pascal coefficients), re-indexed onto the staggered j-line of uᵢ. Their
# stencil reach is ±1 (P=1), {+1,-2} (P=2), {+2,-3} (P=3), {+3,-4} (P=4). The
# array has only ONE ghost layer (Ng=N+2), so P≥3 is only correct via periodic
# index wrapping (`_wrapⱼ`); a non-periodic direction with P≥3 errors (P=2 there
# uses the BC-filled ghost as a crude closure for the two boundary faces).
# NOTE (shared with `sgs!`): `udf!` passes the advecting velocity explicitly —
# `u⁰` in the predictor (where `flow.u` is zeroed by `scale_u!`) and the projected
# `u` in the corrector — so fᵈ acts in both RK phases on the same field the
# convective flux uses (the udf-advecting-velocity fix, regression: test_les.jl).

# periodic-wrapped j-index: interior is 2:Nⱼ-1 with period Nⱼ-2 (Nⱼ=Ng[j])
@inline _wrapⱼ(q,perj,Nⱼ) = perj ? 2 + mod(q-2, Nⱼ-2) : q
# uᵢ at j-offset `s` cells from the right cell of face `Ii=CI(I,i)`
@inline _uᵢ(Ii,j,s,u,perj,Nⱼ) = @inbounds u[CIj(j,Ii,_wrapⱼ(Ii[j]+s,perj,Nⱼ))]
# face advection speed |U_face| (smooth surrogate √(U²+ε²) when ε>0, |U| at ε=0)
@inline _aface(i,j,I,u,ε) = sqrt(ϕ(i,CI(I,j),u)^2 + ε^2)

# Σₚ βₚ Δ^{2p}ⱼ uᵢ at face Ii=CI(I,i), Val-dispatched on P for type stability
@inline _dissip(Ii,j,u,β,::Val{1},perj,Nⱼ) = @inbounds β[1]*(_uᵢ(Ii,j,0,u,perj,Nⱼ)-_uᵢ(Ii,j,-1,u,perj,Nⱼ))
@inline function _dissip(Ii,j,u,β,::Val{2},perj,Nⱼ)
    @inbounds begin
        um2=_uᵢ(Ii,j,-2,u,perj,Nⱼ); um1=_uᵢ(Ii,j,-1,u,perj,Nⱼ); u0=_uᵢ(Ii,j,0,u,perj,Nⱼ); up1=_uᵢ(Ii,j,1,u,perj,Nⱼ)
        return β[1]*(u0-um1) + β[2]*(up1-3u0+3um1-um2)
    end
end
@inline function _dissip(Ii,j,u,β,::Val{3},perj,Nⱼ)
    @inbounds begin
        um3=_uᵢ(Ii,j,-3,u,perj,Nⱼ); um2=_uᵢ(Ii,j,-2,u,perj,Nⱼ); um1=_uᵢ(Ii,j,-1,u,perj,Nⱼ)
        u0=_uᵢ(Ii,j,0,u,perj,Nⱼ); up1=_uᵢ(Ii,j,1,u,perj,Nⱼ); up2=_uᵢ(Ii,j,2,u,perj,Nⱼ)
        return β[1]*(u0-um1) + β[2]*(up1-3u0+3um1-um2) + β[3]*(up2-5up1+10u0-10um1+5um2-um3)
    end
end
@inline function _dissip(Ii,j,u,β,::Val{4},perj,Nⱼ)
    @inbounds begin
        um4=_uᵢ(Ii,j,-4,u,perj,Nⱼ); um3=_uᵢ(Ii,j,-3,u,perj,Nⱼ); um2=_uᵢ(Ii,j,-2,u,perj,Nⱼ); um1=_uᵢ(Ii,j,-1,u,perj,Nⱼ)
        u0=_uᵢ(Ii,j,0,u,perj,Nⱼ); up1=_uᵢ(Ii,j,1,u,perj,Nⱼ); up2=_uᵢ(Ii,j,2,u,perj,Nⱼ); up3=_uᵢ(Ii,j,3,u,perj,Nⱼ)
        return β[1]*(u0-um1) + β[2]*(up1-3u0+3um1-um2) + β[3]*(up2-5up1+10u0-10um1+5um2-um3) +
               β[4]*(up3-7up2+21up1-35u0+35um1-21um2+7um3-um4)
    end
end

"""
    dissipative_flux!(flow, u, t; β, ε=0)

User-defined function (`udf`) adding the implicit-LES parametrized dissipative
numerical flux `fᵈ = -½|U_face| Σₚ βₚ Δ^{2p}ⱼ uᵢ` to the momentum RHS, with the
energy-conserving `cds` convection as the base. `u` is the advecting velocity field
supplied by `udf!` (`u⁰` in the predictor, projected `u` in the corrector). `β` is the vector of `P=length(β)`
dissipation weights (P=1..4); `ε>0` selects a smooth `√(U²+ε²)` surrogate for the
`|U_face|` advection speed (use for exact ForwardDiff gradient reproducibility).

Per-order dissipative sign is `sign(βₚ)=(-1)^(p+1)` (see header): β₁,β₃>0 and
β₂,β₄<0 remove energy. Set the energy-conserving `cds` base at construction
(a `λ` kwarg to `sim_step!` is silently ignored — the scheme lives in `flow.λ`):
    `sim = Simulation(...; λ=cds, ...); sim_step!(sim; udf=dissipative_flux!, β=Float32[β₁,...])`.
β rides the `sim_step!` keyword arguments. Do NOT combine with `quick`/`vanLeer`
or `sgs!` (their dissipation double-counts and contaminates β); a one-time warning
fires if `flow.λ !== cds`.

ForwardDiff w.r.t. β: the flow buffers must carry the `Dual` type, so reconstruct
the `Flow`/`Simulation` with `T=eltype(β)` (as in the Burgers reference) before the
gradient pass — seeding a `Dual` β into a `Float32`/`Float64` `Flow` errors when
storing Duals into `flow.σ`/`flow.f`. (`ε>0` avoids the `√` kink at `U_face=0`.)
"""
function dissipative_flux!(flow, u, t; β, ε=zero(eltype(u)), kwargs...)
    # guard against the silent quick+β double-dissipation (post-#301 a `λ` kwarg to
    # sim_step! is swallowed by the udf kwargs); hasproperty keeps duck-typed
    # (NamedTuple) flows in the AD tests working
    hasproperty(flow, :λ) && flow.λ !== cds &&
        @warn "dissipative_flux! expects the energy-conserving `cds` base but flow.λ=$(flow.λ); build the Simulation/Flow with λ=cds or β also compensates the base scheme's own dissipation" maxlog=1
    _apply_dissipative_flux!(flow.f, flow.σ, u, β, flow.perdir, ε)
end
# core kernel, separated so it can be driven on Dual-typed arrays directly (AD tests)
function _apply_dissipative_flux!(f, σ, u, β, perdir, ε=zero(eltype(u)))
    N,n = size_u(u); P = length(β); valP = Val(P)
    βS = SVector(ntuple(p -> @inbounds(β[p]), valP)) # isbits ⇒ valid GPU kernel arg; preserves eltype (incl. Dual)
    εT = eltype(u)(ε)                                # keep the kernel in the field eltype (no Float64 on GPU)
    for i ∈ 1:n, j ∈ 1:n
        perj = j in perdir
        (P ≥ 3 && !perj) && error("dissipative_flux!: P=$P (Δ^$(2P)) needs ≥$P ghost layers; only supported on periodic directions (non-periodic j=$j). Use P≤2 or widen the halo.")
        # interior j-faces (3:Nⱼ-1)
        @loop (σ[I] = -_aface(i,j,I,u,εT)*_dissip(CI(I,i),j,u,βS,valP,perj,N[j])/2; # /2 keeps eltype
               f[I,i] += σ[I]) over I ∈ inside_u(N,j)
        @loop f[I-δ(j,I),i] -= σ[I] over I ∈ inside_u(N,j)
        # periodic seam face (j-index 2): close the telescoping exactly as
        # conv_diff!'s lower/upperBoundary! do, so cells 2 and Nⱼ-1 get fᵈ too.
        if perj
            @loop (σ[I] = -_aface(i,j,I,u,εT)*_dissip(CI(I,i),j,u,βS,valP,perj,N[j])/2;
                   f[I,i] += σ[I]) over I ∈ slice(N,2,j,2)
            @loop f[I-δ(j,I),i] -= σ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)
        end
    end
end

"""
    spatial_energy_rate(flow; λ=cds, ν=0, udf=nothing, kwargs...)

Discrete kinetic-energy production rate `Σᵢ Σ_cells uᵢ·(duᵢ/dt)` of the SPATIAL
operator only (convection `conv_diff!` plus optional `udf`), summed over one
periodic interior. Diagnostic for the implicit-LES base/dissipation: with
`λ=cds`, `ν=0` and a discretely div-free periodic field it returns ≈0 (the EC
property); `λ=quick` returns a negative O(1) value (limiter dissipation); adding
`udf=dissipative_flux!` returns `≈ -½ Σ_faces |U_face| β₁ (Δuᵢ)²  ≤ 0` (the added
dissipation). Mutates `flow.f` and `flow.σ`.
"""
function spatial_energy_rate(flow; λ=cds, ν=zero(eltype(flow.u)), udf=nothing, kwargs...)
    conv_diff!(flow.f, flow.u, flow.σ, λ; ν, perdir=flow.perdir)
    # static diagnostic ⇒ advecting field is flow.u; udf takes the (flow,u,t) signature
    isnothing(udf) || udf(flow, flow.u, zero(eltype(flow.u)); kwargs...)
    N,_ = size_u(flow.u); R = inside_u(N)
    sum(@view(flow.u[R]) .* @view(flow.f[R]))   # broadcast+reduce (GPU-safe; no scalar indexing)
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

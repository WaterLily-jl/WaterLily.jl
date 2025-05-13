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
@fastmath @inline function dot(a,b)
    init=zero(eltype(a))
    @inbounds for ij in eachindex(a)
     init += a[ij] * b[ij]
    end
    return init
end

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

using LinearAlgebra: eigvals, Hermitian
"""
    λ₂(I::CartesianIndex{3},u)

λ₂ is a deformation tensor metric to identify vortex cores.
See [https://en.wikipedia.org/wiki/Lambda2_method](https://en.wikipedia.org/wiki/Lambda2_method) and
Jeong, J., & Hussain, F., doi:[10.1017/S0022112095000462](https://doi.org/10.1017/S0022112095000462)
"""
function λ₂(I::CartesianIndex{3},u)
    J = @SMatrix [∂(i,j,I,u) for i ∈ 1:3, j ∈ 1:3]
    S,Ω = (J+J')/2,(J-J')/2
    eigvals(Hermitian(S^2+Ω^2))[2]
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
    θ = z × (loc(0,I,eltype(u))-SVector{3}(center))
    n = norm2(θ)
    n<=eps(n) ? 0. : θ'*ω(I,u) / n
end

"""
    nds(body,x,t)

BDIM-masked surface normal.
"""
@inline function nds(body,x,t)
    d,n,_ = measure(body,x,t,fastd²=1)
    n*WaterLily.kern(clamp(d,-1,1))
end

"""
    pressure_force(sim::Simulation)

Compute the pressure force on an immersed body.
"""
pressure_force(sim) = pressure_force(sim.flow,sim.body)
pressure_force(flow,body) = pressure_force(flow.p,flow.f,body,time(flow))
function pressure_force(p,df,body,t=0)
    Tp = eltype(p); To = promote_type(Float64,Tp)
    df .= zero(Tp)
    @loop df[I,:] .= p[I]*nds(body,loc(0,I,Tp),t) over I ∈ inside(p)
    sum(To,df,dims=ntuple(i->i,ndims(p)))[:] |> Array
end

"""
    S(I::CartesianIndex,u)

Rate-of-strain tensor.
"""
S(I::CartesianIndex{2},u) = @SMatrix [0.5*(∂(i,j,I,u)+∂(j,i,I,u)) for i ∈ 1:2, j ∈ 1:2]
S(I::CartesianIndex{3},u) = @SMatrix [0.5*(∂(i,j,I,u)+∂(j,i,I,u)) for i ∈ 1:3, j ∈ 1:3]
"""
   viscous_force(sim::Simulation)

Compute the viscous force on an immersed body.
"""
viscous_force(sim) = viscous_force(sim.flow,sim.body)
viscous_force(flow,body) = viscous_force(flow.u,flow.ν,flow.f,body,time(flow))
function viscous_force(u,ν,df,body,t=0)
    Tu = eltype(u); To = promote_type(Float64,Tu)
    df .= zero(Tu)
    @loop df[I,:] .= -2ν*S(I,u)*nds(body,loc(0,I,Tu),t) over I ∈ inside_u(u)
    sum(To,df,dims=ntuple(i->i,ndims(u)-1))[:] |> Array
end

"""
   total_force(sim::Simulation)

Compute the total force on an immersed body.
"""
total_force(sim) = pressure_force(sim) .+ viscous_force(sim)

using LinearAlgebra: cross
"""
    pressure_moment(x₀,sim::Simulation)

Computes the pressure moment on an immersed body relative to point x₀.
"""
pressure_moment(x₀,sim) = pressure_moment(x₀,sim.flow,sim.body)
pressure_moment(x₀,flow,body) = pressure_moment(x₀,flow.p,flow.f,body,time(flow))
function pressure_moment(x₀,p,df,body,t=0)
    Tp = eltype(p); To = promote_type(Float64,Tp)
    df .= zero(Tp)
    @loop df[I,:] .= p[I]*cross(loc(0,I,Tp)-x₀,nds(body,loc(0,I,Tp),t)) over I ∈ inside(p)
    sum(To,df,dims=ntuple(i->i,ndims(p)))[:] |> Array
end

"""
     MeanFlow{T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Mf<:AbstractArray{T}}

Holds temporal averages of velocity, squared velocity, pressure, and Reynolds stresses.
"""
struct MeanFlow{T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Mf<:AbstractArray{T}}
    P :: Sf # pressure scalar field
    U :: Vf # velocity vector field
    UU :: Union{Mf,Nothing} # squared velocity tensor, u_i(x)u_j
    t :: Vector{T} # time steps vector
    uu_stats :: Bool # flag to compute UU on-the-fly temporal averages
    function MeanFlow(flow::Flow{D,T}; t_init=time(flow), uu_stats=false) where {D,T}
        f = typeof(flow.u).name.wrapper
        P = zeros(T, size(flow.p)) |> f
        U = zeros(T, size(flow.u)) |> f
        UU = uu_stats ? zeros(T, size(flow.p)...,D,D) |> f : nothing
        new{T,typeof(P),typeof(U),typeof(UU)}(P,U,UU,T[t_init],uu_stats)
    end
end

time(meanflow::MeanFlow) = meanflow.t[end]-meanflow.t[1]

function reset!(meanflow::MeanFlow; t_init=0.0)
    fill!(meanflow.P, 0); fill!(meanflow.U, 0)
    !isnothing(meanflow.UU) && fill!(meanflow.UU, 0)
    deleteat!(meanflow.t, collect(1:length(meanflow.t)))
    push!(meanflow.t, t_init)
end

function update!(meanflow::MeanFlow, flow::Flow)
    dt = time(flow) - meanflow.t[end]
    ε = dt / (dt + (meanflow.t[end] - meanflow.t[1]) + eps(eltype(flow.p)))
    @loop meanflow.P[I] = ε * flow.p[I] + (1 - ε) * meanflow.P[I] over I in CartesianIndices(flow.p)
    @loop meanflow.U[Ii] = ε * flow.u[Ii] + (1 - ε) * meanflow.U[Ii] over Ii in CartesianIndices(flow.u)
    if meanflow.uu_stats
        for i in 1:ndims(flow.p), j in 1:ndims(flow.p)
            @loop meanflow.UU[I,i,j] = ε * (flow.u[I,i] .* flow.u[I,j]) + (1 - ε) * meanflow.UU[I,i,j] over I in CartesianIndices(flow.p)
        end
    end
    push!(meanflow.t, meanflow.t[end] + dt)
end

uu!(τ,a::Meanflow) = for i in 1:ndims(a.P), j in 1:ndims(a.P)
    @loop τ[I,i,j] = a.UU[I,i,j] - a.U[I,i,j] * a.U[I,i,j] over I in CartesianIndices(a.P)
end
function uu(a::Meanflow)
    τ = zeros(eltype(a.UU), size(a.UU)...) |> typeof(meanflow.UU).name.wrapper
    uu!(τ,a)
    return τ
end

function copy!(a::Flow, b::MeanFlow)
    a.u .= b.U
    a.p .= b.P
end
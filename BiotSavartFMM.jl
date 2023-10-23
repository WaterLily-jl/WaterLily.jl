using WaterLily
using LinearAlgebra
include("examples/TwoD_plots.jl")


# function show_multi(a::BiotSavart)
#     ins = WaterLily.inside(a.ω[:,:,1])
#     uniq = unique(.√sum(a.ω[ins,:].^2,dims=3))
#     color = zero(a.ω[ins,1])
#     for it in 1:length(uniq)
#         Is = findall(.√sum(a.ω[ins,:].^2,dims=3).==uniq[it])
#         color[Is] .= it
#     end
#     return color
# end


abstract type AbstractBiotSavart{T,V} end

struct BiotSavart{T,V<:AbstractArray{T}} <: AbstractBiotSavart{T,V}
    ω::V
    r::V
    function BiotSavart(ω::AbstractArray{T},r::AbstractArray{T}) where T
        new{T,typeof(ω)}(ω,r)
    end
end

@inline up(I::CartesianIndex,a=0) = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,I))
@inline down(I::CartesianIndex) = WaterLily.CI((I+2oneunit(I)).I .÷2)
@fastmath @inline function downsample(I::CartesianIndex,b)
    s = zero(eltype(b))
    for J ∈ up(I)
     s += @inbounds(b[J])
    end
    return s
end

function downsample!(a,b)
    Na,n = WaterLily.size_u(a)
    for i ∈ 1:n
        @WaterLily.loop a[I,i] = downsample(I,b[:,:,i]) over I ∈ CartesianIndices(map(n->2:n-1,Na))
    end
end

function upsample!(a,b)
    Na,n = WaterLily.size_u(a)
    for i ∈ 1:n
        @WaterLily.loop a[I,i] = b[down(I),i] over I ∈ WaterLily.inside(a[:,:,i])
    end
end


function downsample(b::BiotSavart)
    N,n = WaterLily.size_u(b.ω)
    Na = map(i->1+i÷2,N)
    aω = similar(b.ω,(Na...,n)); fill!(aω,0)
    ar = similar(b.r,(Na...,n)); fill!(ar,0)
    downsample!(ar,b.r); ar./=4
    BiotSavart(aω,ar)
end


@inline divisible(N) = mod(N,2)==0 && N>4

struct MultiLevelBiotSavart{T,V<:AbstractArray{T}} <: AbstractBiotSavart{T,V}
    ω::V
    r::V
    levels :: Vector{BiotSavart{T,V}}
    function MultiLevelBiotSavart(ω::AbstractArray{T},maxlevels=6) where T
        r = similar(ω); apply!((i,x)->x[i],r)
        levels = BiotSavart[BiotSavart(ω,r)]
        while all(size(levels[end].ω[:,:,1]) .|> divisible) && length(levels) <= maxlevels
            push!(levels,downsample(levels[end]))
        end
        text = "MultiLevelBiotSavart requires size=a2ⁿ, where n>2"
        @assert (length(levels)>2) text
        new{T,typeof(ω)}(ω,r,levels)
    end
end


function BiotSavart!(x::Vector{T},mlb::MultiLevelBiotSavart;l=1) where T
    fine,coarse = mlb.levels[l],mlb.levels[l+1]
    if l+1<length(mlb.levels) # a level below exist, update it first
        BiotSavart!(x,mlb,l=l+1)
        upsample!(fine.ω,coarse.ω)
    else # no level below, update the finest level
        Is = WaterLily.inside(coarse.ω[:,:,1])
        BiotSavartKernel!(x,coarse,Is)
        upsample!(fine.ω,coarse.ω)
    end
    # update this level
    Is = WaterLily.inside(fine.ω[:,:,1])
    # threshold for refinement 
    half_width = (maximum(fine.r[:,:,1])-minimum(fine.r[:,:,1]))/2
    r = copy(fine.r); r[:,:,1] .-= x[1]; r[:,:,2] .-= x[2]
    r = .√sum(r.^2,dims=3)
    Is = Is[r[Is].≤√2*half_width]
    BiotSavartKernel!(x,fine,Is)
end
"""
Flags cells where a refinement is required
"""
function refine(a)
    Is = WaterLily.inside(a.ω[:,:,1])
    # threshold for refinement 
    half_width = (maximum(a.r[:,:,1])-minimum(a.r[:,:,1]))/2
    r = a.r; r[:,:,1] .=x[1]
    Is[[Is].≤half_width]
end

"""
Biot-Savart kernel in the `i`-th direction at `x` due to vorticity at cell edge `b[J,:]`
"""
function BiotSavartKernel!(x::Vector{Float64},l::BiotSavart,Is,ϵ=1e-6)
    N,n = WaterLily.size_u(l.ω)
    for i ∈ 1:n, J ∈ Is
        j = i%2+1
        r = x .- l.r[J,:] # cell edge vorticity
        rⁿ = norm(r)^2
        l.ω[J,i] = sign(i-j)*r[j]/(2π*rⁿ+ϵ^2)
    end
end

"""
RankineVortex(i,xy,center,R,Γ)
"""
function RankineVortex(i, xy, center, R=4, Γ=1)
    xy = (xy .- 1.5 .- center)
    x,y = xy
    θ = atan(y,x)
    r = norm(xy)
    vθ =Γ/2π*(r<=R ? r/R^2 : 1/r)
    v = [-vθ*sin(θ),vθ*cos(θ)]
    return v[i]
end


# some definitons
U = 1
Re = 250
m, n = 2^6, 2^6

# make a simulation
sim = Simulation((n,m), (U,0), m; ν=U*m/Re, T=Float64)
u = copy(sim.flow.u)

# make a Rankine vortex
f(i,x) = RankineVortex(i,x,(m/2,m/2),10, 1)

# apply it to the flow
apply!(f, sim.flow.u)

flood(sim.flow.u[:,:,1]; shift=(-0.5,-0.5))
flood(sim.flow.u[:,:,2]; shift=(-0.5,-0.5))

# compute vorticity
@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)
flood(sim.flow.σ; shift=(-0.5,-0.5))

# multi level biot savart
BioSavart = MultiLevelBiotSavart(zero(sim.flow.u))
N,n = WaterLily.size_u(sim.flow.u)
u = zero(sim.flow.u);

# method where we pass it a empty kernel array a a position array
for Is ∈ WaterLily.inside(sim.flow.σ)
    # using cell center doesn't change this much
    BiotSavart!(Vector(loc(0,Is)),BioSavart)
    for i ∈ 1:n
        u[Is,i] = dot(BioSavart.levels[1].ω[:,:,i],sim.flow.σ)
    end
end

# error
BC!(sim.flow.u,zeros(2))
BC!(u,zeros(2))
println("L₂-norm error u-velocity ", WaterLily.L₂(u[:,:,1].-sim.flow.u[:,:,1]))
println("L₂-norm error v-velocity ", WaterLily.L₂(u[:,:,2].-sim.flow.u[:,:,2]))
flood(u[:,:,1].-sim.flow.u[:,:,1])

# # FMM kernel
# L₂-norm error u-velocity 2.716044361118482e-6
# L₂-norm error v-velocity 2.7160443611184675e-6

# # standard kernel error
# L₂-norm error u-velocity 1.0034365372479126e-6
# L₂-norm error v-velocity 1.0034365372479018e-6


# N = (66,66)
# n = 2
# ω0 = zeros((N...,n)); #apply!((i,x)->1,ω0)
# ω0[[1,end],:,:] .= 0;
# ω0[:,[1,end],:] .= 0;
# display(ω0)

# mlb = MultiLevelBiotSavart(ω0)
# x = [0.,0.]
# BiotSavart!(x,mlb)

# for i ∈ 1:length(mlb.levels)
#     println("Level ", i)
#     display(mlb.levels[i].ω[:,:,1])
#     display(mlb.levels[i].r[:,:,1])
#     println("total vort. ", sum(mlb.levels[i].ω[:,:,1]),"\n")
# end

# # try upsample one layer layer
# upsample!(mlb.levels[2].r,mlb.levels[3].r)
# display(mlb.levels[3].r[:,:,1])
# display(mlb.levels[2].r[:,:,1])

# update!(mlb,[0.,0.])
# flood(mlb.levels[1].ω[:,:,1]; shift=(-0.5,-0.5))

# try MultiPol
# println()
# println("Try multigrid")
# x =[0.0,0.0]
# Is = WaterLily.inside(mlb.levels[5].ω[:,:,1])
# println("First level")
# BiotSavartKernel!(x,mlb.levels[5],Is)
# flood(mlb.levels[5].ω[:,:,1]; shift=(-0.5,-0.5))

# # upsample the distance and the kernel
# upsample!(mlb.levels[4].ω,mlb.levels[5].ω)
# flood(mlb.levels[4].ω[:,:,1]; shift=(-0.5,-0.5))

# # decide where to refine
# r = .√(sum(mlb.levels[4].r.^2,dims=3)[:,:,1])
# insi = WaterLily.inside(r)
# r = r[insi]
# Is = insi[r.<=(maximum(r)-minimum(r))/2]
# BiotSavartKernel!(x,mlb.levels[4],Is)
# flood(mlb.levels[4].ω[:,:,1]; shift=(-0.5,-0.5))

# # test biot Savart
# println()
# println("Try multigrid")
# x =[0.0,0.0]
# a = mlb.levels[3].ω[:,:,1]
# Is = CartesianIndices(a)[a.>= 0.0,1]
# println("First level")
# BiotSavart!(x,mlb.levels[3].r,mlb.levels[3].ω,Is)
# display(mlb.levels[3].r[:,:,1])
# display(mlb.levels[3].ω[:,:,1])

# println("updample one level")
# upsample!(mlb.levels[2].ω,mlb.levels[3].ω)
# display(mlb.levels[3].ω[:,:,1])
# display(mlb.levels[2].ω[:,:,1])

# println("Second level")
# a = mlb.levels[2].ω[:,:,1]
# Is  = CartesianIndices(a)[a.>= 0.05,1]
# # level above
# BiotSavart!(x,mlb.levels[2].r,mlb.levels[2].ω,Is)
# display(mlb.levels[2].r[:,:,1])
# display(mlb.levels[2].ω[:,:,1])


# println("updample one level")
# upsample!(mlb.levels[1].ω,mlb.levels[2].ω)
# display(mlb.levels[2].ω[:,:,1])
# display(mlb.levels[1].ω[:,:,1])

# println("Second level")
# a = mlb.levels[1].ω[:,:,1]
# Is  = CartesianIndices(a)[a.>= 0.1,1]
# # level above
# BiotSavart!(x,mlb.levels[1].r,mlb.levels[1].ω,Is)
# display(mlb.levels[1].r[:,:,1])
# display(mlb.levels[1].ω[:,:,1])



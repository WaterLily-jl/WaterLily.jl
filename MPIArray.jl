using WaterLily,StaticArrays
using KernelAbstractions

struct MPIArray{T,N,V<:AbstractArray{T,N},W<:AbstractVector{T}} <: AbstractArray{T,N}
    A :: V
    send :: W
    recv :: W
    function MPIArray(::Type{T}, dims::NTuple{N, Integer}) where {T,N}
        A = Array{T,N}(undef, dims); fill!(A, zero(T))
        send, recv = zeros(T,maximum(dims)), zeros(T,maximum(dims))
        new{T,N,typeof(A),typeof(send)}(A,send,recv)
    end
    MPIArray(A::AbstractArray{T}) where T = MPIArray(T,size(A))
end
for fname in (:size, :length, :ndims, :eltype) # how to write 4 lines of code in 5...
    @eval begin
        Base.$fname(A::MPIArray) = Base.$fname(A.A)
    end
end
Base.getindex(A::MPIArray, i...)            = Base.getindex(A.A, i...)
Base.setindex!(A::MPIArray, v, i...)        = Base.setindex!(A.A, v, i...)
Base.copy(A::MPIArray) = (B=MPIArray(eltype(A),size(A)); B.A.=A.A; B)
Base.similar(A::MPIArray)                   = MPIArray(eltype(A),size(A))
Base.similar(A::MPIArray, dims::Tuple)      = MPIArray(eltype(A),dims)
# required for the @loop function
KernelAbstractions.get_backend(A::MPIArray) = KernelAbstractions.get_backend(A.A)

# # initialize array
Nd = (10,10,2)
u = Array{Float64}(undef, Nd...) |> MPIArray
p = zeros(Base.front(Nd)...) |> MPIArray

# circle simulation
function circle(n,m;Re=250,U=1)
    radius, center = m/8, m/2
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, mem=MPIArray)
end

# test on sim
include("examples/TwoD_plots.jl")
sim = circle(3*2^6,2^7)
sim_gif!(sim,duration=10,clims=(-5,5),plotbody=true)

# @btime mom_step!($sim.flow,$sim.pois)      # with MPIArray
#   5.015 ms (24899 allocations: 1.87 MiB)
# @btime mom_step!($sim.flow,$sim.pois)      # with Array
#   2.649 ms (22285 allocations: 1.55 MiB)
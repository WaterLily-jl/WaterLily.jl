module WaterLilyHYPREExt

if isdefined(Base, :get_extension)
    using HYPRE
else
    using ..HYPRE
end

using SparseArrays
using WaterLily
import WaterLily: inside,size_u,@loop
import WaterLily: AbstractPoisson,solver!,update!,HyprePoisson,putback!,L₂,L∞

# @inline insidep(a::AbstractArray,perdir=()) = CartesianIndices(ntuple( i-> i ∈ perdir ? (1:size(a,i)) : (2:size(a,i)-1), length(axes(a))))

# @fastmath @inline function filldiag(I::CartesianIndex{d},x,L,perdir) where {d}
#     s = zero(eltype(L))
#     for i in 1:d
#         Ip = I+δ(i,I)
#         (i ∈ perdir && Ip ∉ insidep(x,perdir)) && (Ip = WaterLily.CIj(i,Ip,1))
#         s -= @inbounds(L[I,i]+L[Ip,i])
#     end
#     return s
# end

struct HYPREPoisson{T,V<:AbstractVector{T},M<:AbstractArray{T},
                    Vf<:AbstractArray{T},Mf<:AbstractArray{T},
                    SO<:HYPRE.HYPRESolver} <: AbstractPoisson{T,Vf,Mf}
    ϵ::V   # Hypre.Vector (increment)
    A::M   # Hypre.SparseMatrixCSC
    r::V   # Hypre.Vector (residual)
    x::Vf  # WaterLily approximate solution
    L::Mf  # WaterLily lower diagonal coefficients
    z::Vf  # WaterLily source
    solver::SO
end
function HyprePoisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};
                      MaxIter=1000, Tol=100eps(T), PrintLevel=0, Logging=0, 
                      Precond=HYPRE.BoomerAMG(), perdir=()) where T
    @assert T==Float64 "Only Float64 are supported by HYPRE"
    # create the vectors and the matrix
    ϵ = Vector{T}(undef, prod(size(x)))
    r = Vector{T}(undef, prod(size(z)))
    A = SparseMatrixCSC{T,Int}(undef, prod(size(x)), prod(size(x)))
    J = LinearIndices(x) # useful
    N,n = WaterLily.size_u(L)
    for I in CartesianIndices(x) # fill the vectors entirely
        ϵ[J[I]] = x[I]; r[J[I]] = z[I]
        A[J[I],J[I]] = one(T) # set all diagonals to one
    end
    for I in inside(x)
        A[J[I],J[I]] = WaterLily.diag(I,L)
        for i in 1:n # fill diagonal terms
            Im = I-δ(i,I); Ip = I+δ(i,I)
            A[J[Im],J[I]] = L[I,i]  # x[I-δ(i,I)]*L[I,i]
            A[J[Ip],J[I]] = L[Ip,i] # x[I+δ(i,I)]*L[I+δ(i,I),i]
        end
    end
    # if we have a purely Neumann problem, we must fix the pressure at one spot to satisfiability
    # the system, we choose the first block
    # length(perdir) == n && (A[1,1] = one(T); A[1,2:end] .= zero(T);
    #                         A[1,1] = one(T); A[1,2:end] .= zero(T))
    HYPRE.Init() # Init and create a conjugate gradients solver
    solver = HYPRE.GMRES(;MaxIter,Tol,PrintLevel,Logging,Precond)
    return HYPREPoisson(ϵ,A,r,x,L,z,solver)
end
export HYPREPoisson

"""
Fill back the solution `hp.x` and `hp.z` from the solution `hp.e` and `hp.r`.
"""
function putback!(hp::HYPREPoisson)
    J = LinearIndices(hp.x)
    @loop (hp.x[I] = hp.ϵ[J[I]]; hp.z[I] = hp.r[J[I]]) over I ∈ CartesianIndices(hp.x)
end
function Base.fill!(hp::HYPREPoisson)
    J = LinearIndices(hp.x)
    @loop (hp.ϵ[J[I]] = hp.x[I]; hp.r[J[I]] = hp.z[I]) over I ∈ CartesianIndices(hp.x)
end
update!(hp::HYPREPoisson) = nothing
"""
Solve the poisson problem and puts back the solution where it is expected.
"""
function solver!(hp::HYPREPoisson;kwargs...)
    fill!(hp)
    HYPRE.solve!(hp.solver, hp.ϵ,hp.A, hp.r)
    putback!(hp); 
    # scale the solution for zero mean
    mean = @inbounds(sum(hp.x[inside(hp.x)]))/length(inside(hp.x))
    @loop hp.x[I] -= mean over I ∈ inside(hp.x)
    J = LinearIndices(hp.x)
    @loop hp.ϵ[J[I]] -= mean over I ∈ inside(hp.x) # don't remove ean frome ghost cells
    return nothing
end

L₂(p::HYPREPoisson) = sum(abs2,p.A*p.ϵ.-p.r)
L∞(p::HYPREPoisson) = maximum(abs,p.A*p.ϵ.-p.r)

end # module
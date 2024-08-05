using HYPRE
using SparseArrays
using LinearAlgebra
using WaterLily; δ
using WaterLily
using Revise
include("examples/TwoD_plots.jl")


# struct PrecondConjugateGradient{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
#     L :: V # Lower diagonal coefficients
#     D :: S # Diagonal coefficients
#     iD :: S # 1/Diagonal
#     x :: S # approximate solution
#     ϵ :: S # increment/error
#     r :: S # residual
#     z :: S # source
#     n :: Vector{Int16} # pressure solver iterations
#     perdir :: NTuple # direction of periodic boundary condition
#     function PrecondConjugateGradient(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};perdir=()) where T
#         @assert axes(x) == axes(z) && axes(x) == Base.front(axes(L)) && last(axes(L)) == eachindex(axes(x))
#         r = similar(x); fill!(r,0)
#         ϵ,D,iD = copy(r),copy(r),copy(r)
#         WaterLily.set_diag!(D,iD,L)
#         new{T,typeof(x),typeof(L)}(L,D,iD,x,ϵ,r,z,[],perdir)
#     end
# end
# function CholeskyPreconditioner(A)
#     L = copy(A)
#     @inbounds for j in 1:size(L, 2)
#         d = sqrt(L[j,j])
#         L[j,j] = d
#         for i in Base.Iterators.drop(nzrange(L,j), 1)
#             L *= 1.0/d
#         end
#     end
#     return L
# end


struct HYPREPoisson{T,V<:AbstractVector{T},M<:AbstractArray{T},
                    Vf<:AbstractArray{T},Mf<:AbstractArray{T},
                    SO<:HYPRE.HYPRESolver} <: AbstractPoisson{T,Vf,Mf}
    ϵ::V   # Hypre.Vector (increment/error)
    A::M   # Hypre.SparseMatrixCSC
    r::V   # Hypre.Vector (residual)
    x::Vf  # WaterLily approximate solution
    L::Mf  # WaterLily lower diagonal coefficients
    z::Vf  # WaterLily source
    solver::SO
    function HYPREPoisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};
                          MaxIter=1000, Tol=1e-9, PrintLevel=0, Logging=0, 
                          Precond=HYPRE.BoomerAMG(), perdir=()) where T
        # create the vectors and the matrix
        ϵ = Vector{T}(undef, prod(size(x)))
        r = Vector{T}(undef, prod(size(z)))
        A = SparseMatrixCSC{T,Int}(undef, prod(size(x)), prod(size(x)))
        J = LinearIndices(x) # useful
        for I in CartesianIndices(x) # fill the vectors entirely
            ϵ[J[I]] = x[I]; r[J[I]] = z[I]
        end
        # ϵ .= reduce(vcat, @views(x[:])); r .= reduce(vcat, @views(z[:]))
        for I in inside(x) # fill inside the domain only otherwise we will go outside with ±δ
            A[J[I],J[I]] = WaterLily.diag(I,L)
            for i in 1:last(size(L)) # fill diagonal terms
                A[J[I-δ(i,I)],J[I]] = L[I,i]        # x[I-δ(i,I)]*L[I,i]
                A[J[I+δ(i,I)],J[I]] = L[I+δ(i,I),i] # x[I+δ(i,I)]*L[I+δ(i,I),i]
            end
        end
        zero_idx = []
        for i ∈ 1:size(A,1)
            all(A[i,:].≈0) && push!(zero_idx,i)
        end
        idx = collect(1:size(A,1))
        deleteat!(idx,zero_idx); deleteat!(ϵ,zero_idx); deleteat!(r,zero_idx)
        A_ = copy(A[idx,idx])
        HYPRE.Init() # Init and create a conjugate gradients solver
        solver = HYPRE.GMRES(;MaxIter,Tol,PrintLevel,Logging,Precond)
        new{T,typeof(ϵ),typeof(A_),typeof(x),typeof(L),typeof(solver)}(ϵ,A_,r,x,L,z,solver)
    end
end
export HYPREPoisson
"""
Fill back the solution `hp.x` and `hp.z` from the solution `hp.e` and `hp.r`.
"""
function putback!(hp::HYPREPoisson)
    # J = LinearIndices(hp.r)
    # for I in CartesianIndices(hp.x)
    #     hp.x[I] = hp.ϵ[J[I]]; hp.z[I] = hp.r[J[I]]
    # end
    hp.x .= 0.; hp.z .= 0.
    hp.x[inside(hp.x)] .= reshape(hp.ϵ,size(hp.x[inside(hp.x)]));
    hp.z[inside(hp.z)] .= reshape(hp.r,size(hp.z[inside(hp.z)]));
end
function Base.fill!(hp::HYPREPoisson)
    # J = LinearIndices(hp.r)
    # for I in CartesianIndices(hp.x) # fill the vectors entirely
    #     hp.ϵ[J[I]] = hp.x[I]; hp.r[J[I]] = hp.z[I]
    # end
    hp.ϵ .= reduce(vcat, hp.x[inside(hp.x)])
    hp.r .= reduce(vcat, hp.z[inside(hp.z)])
    for I in inside(hp.x) # fill the vectors entirely
        hp.ϵ[J[I]] = hp.x[I]; hp.r[J[I]] = hp.z[I]
    end
end
update!(hp::HYPREPoisson) = nothing
"""
Solve the poisson problem and puts back the solution where it is expected.
"""
function WaterLily.solver!(hp::HYPREPoisson;kwargs...)
    fill!(hp)
    HYPRE.solve!(hp.solver, hp.ϵ, hp.A, hp.r)
    putback!(hp); 
    # scale the solution for zero mean
    mean = @inbounds(sum(hp.x[inside(hp.x)]))/length(inside(hp.x))
    @WaterLily.loop hp.x[I] -= mean over I ∈ inside(hp.x)
    return nothing
end

N = 2^6+2
p = zeros(N,N) #Array{Float64}(reshape(1:N^2,(N,N)))
L = ones(N,N,2); WaterLily.BC!(L,zeros(2))
σ = zeros(N,N)
pois = MultiLevelPoisson(copy(p),copy(L),copy(σ))
hypre = HYPREPoisson(copy(p),copy(L),copy(σ))

# matrix mult does the same, this means the coefficient are at the correct spot
x = Array{Float64}(reshape(1:N^2,(N,N)))
result = WaterLily.mult!(pois,x)
println(pois.z) # same as result
# fill the hypre vectors
hypre.z.=x; fill!(hypre)
hypre.ϵ .= hypre.A*hypre.r
putback!(hypre) # make a field again
println(Matrix(hypre.x))
@show norm(result.-hypre.x)

# hydrostatic pressure test case
hyrostatic!(p) = @WaterLily.inside p.z[I] = WaterLily.∂(1,I,p.L) # zero v contribution everywhere

hyrostatic!(pois)
hyrostatic!(hypre); fill!(hypre)
# solve back
WaterLily.solver!(pois;tol=1e-9)
WaterLily.solver!(hypre)


# # Initialize HYPRE
# HYPRE.Init()

# N = 16
# λ = collect(2 .+ (1:N));
# A = tril(rand(N,N),1) + diagm(λ)
# A = SparseMatrixCSC(A)
# b = Vector(rand(N))

# # Create a conjugate gradients solver
# solver = HYPRE.PCG(; MaxIter = 1000, Tol = 1e-9, 
#                      PrintLevel = 2, Logging = 1, Precond = HYPRE.BoomerAMG())

# # Compute the solution
# x = zeros(length(b))
# @time HYPRE.solve!(solver, x, A, b)
# # @show x
# @show norm(A*x - b)

# WaterLily.solver!(pois::MultiLevelPoisson) = WaterLily.solver!(pois;tol=1e-9)

# # the classic...
function TGV(; pow=6, Re=1000, T=Float64, mem=Array)
    # Taylor-Green-Vortex initial velocity field
    function u_TGV(i,x,t,ν,κ)
        i==1 && return sin(κ*x[1])*cos(κ*x[2])*exp(-2κ^2*ν*t) # u_x
        return  -cos(κ*x[1])*sin(κ*x[2])*exp(-2κ^2*ν*t)       # u_y
    end
    # Define vortex size, velocity, viscosity
    L = 2^pow; U = 1; ν = U*L/Re
    # make the function
    uλ(i,xy) = u_TGV(i,xy,0,ν,2π/L)
    # Initialize simulation
    return Simulation((L,L), (0,0), L; U, uλ, ν, T, mem, psolver=HYPREPoisson)
end

# Initialize and run
sim = TGV()
mom_step!(sim.flow,sim.pois)
sim_gif!(sim,duration=10,clims=(-5,5))


# # # GMRES
# x = zeros(length(b))
# @time IterativeSolvers.gmres!(x, A, b; maxiter=1000, reltol=1e-9)
# @show norm(A*x - b)

# # try CG on a non-symmetric matrix
# N = 16
# A = Tridiagonal(ones(N-1), -2ones(N), ones(N-1))
# A[1,:] .= 0
# A[:,1] .= 0
# A[end,:] .= 0
# A[:,end] .= 0
# A[1,1] = 1
# A[end,end] = 1
# A[N÷ 2,:] .= 0.0
# A[:,N÷ 2] .= 0.0
# A = SparseMatrixCSC(-A)
# b = Vector(sin.(2π.*collect(0:1/(N-1):1)))

# solver = HYPRE.PCG()

# x = zeros(length(b))
# @time HYPRE.solve!(solver, x, A, b)
# @show norm(A*x - b)


 # for I in inside(r)
            # J = LinearIndices(r)[I]
        #     CIs = [CartesianIndex(J,J).I...] # unpack to modify
        #     for i in 1:n
        #         if i==1 # upper and lower in the x-direction -> column major
        #             CIs[i] = CIs[i]+1
        #             upper = CartesianIndex(CIs...)
        #             CIs[i] = CIs[i]-2
        #             lower = CartesianIndex(CIs...)
        #         else # upper and lower in the y-direction, one domain away
        #             CIs[i] = CIs[i]+N[i]
        #             upper = CartesianIndex(CIs...)
        #             CIs[i] = CIs[i]-2N[i]
        #             lower = CartesianIndex(CIs...)
        #         end
        #         # check that they are on the domain
        #         if lower ∈ CartesianIndices(map(_->1:prod(N),N))
        #             A[lower] = L[I,i] # lower
        #         end
        #         if upper ∈ CartesianIndices(map(_->1:prod(N),N))
        #             A[upper] = L[I+δ(i,I),i] # upper
        #         end
        #     end
        # end
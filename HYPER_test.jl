using HYPRE
using SparseArrays
using LinearAlgebra
using WaterLily; δ
using WaterLily
using Revise
include("examples/TwoD_plots.jl")

struct HYPREPoisson{T,V<:AbstractVector{T},M<:AbstractArray{T},
                    Vf<:AbstractArray{T},Mf<:AbstractArray{T},
                    SO<:HYPRE.HYPRESolver} <: AbstractPoisson{T,Vf,Mf}
    x::V   # Hypre.Vector
    A::M   # Hypre.SparseMatrixCSC
    b::V   # Hypre.Vector
    r::Vf  # WaterLily.σ
    L::Mf  # WaterLily.L
    z::Vf  # WaterLily.x
    solver::SO
    function HYPREPoisson(r::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};
                          MaxIter=1000, Tol=1e-9, PrintLevel=2, Logging=1, 
                          Precond=HYPRE.BoomerAMG(), perdir=()) where T
        # create the vectors and the matrix
        x = Vector{T}(undef, prod(size(r)))
        b = Vector{T}(undef, prod(size(z)))
        A = SparseMatrixCSC{T,Int}(undef, prod(size(r)), prod(size(r)))
        J = LinearIndices(r) # useful
        for I in CartesianIndices(r) # fill the vectors entirely
            x[J[I]] = r[I]; b[J[I]] = z[I]
            # A[J[I],J[I]] = eps(T)
        end
        for I in inside(r) # fill inside the domain only otherwise we will go outside with ±δ
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
        A_ = SparseMatrixCSC{T,Int}(undef, size(A,1)-length(zero_idx), size(A,2)-length(zero_idx))
        k=1
        for i ∈ 1:size(A,1)
            if !(i in zero_idx)
                A_[k,:] .= 1 #A[i,setdiff(1:size(A,2),zero_idx)]
                k+=1
            end
        end
        deleteat!(x,zero_idx); deleteat!(b,zero_idx)
        HYPRE.Init() # Init and create a conjugate gradients solver
        solver = HYPRE.PCG(;MaxIter,Tol,PrintLevel,Logging,Precond)
        new{T,typeof(x),typeof(A_),typeof(r),typeof(L),typeof(solver)}(x,A_,b,copy(r),copy(L),copy(z),solver)
    end
end
export HYPREPoisson
"""
Fill back the solution `hp.x` and `hp.z` from the solution `hp.e` and `hp.r`.
"""
function putback!(hp::HYPREPoisson)
    J = LinearIndices(hp.r)
    for I in CartesianIndices(hp.r)
        hp.r[I] = hp.x[J[I]]; hp.z[I] = hp.b[J[I]]
    end
end
function Base.fill!(hp::HYPREPoisson)
    J = LinearIndices(hp.r)
    for I in CartesianIndices(hp.r) # fill the vectors entirely
        hp.x[J[I]] = hp.r[I]; hp.b[J[I]] = hp.z[I]
    end
end
update!(hp::HYPREPoisson) = nothing
"""
Solve the poisson problem and puts back the solution where it is expected.
"""
function WaterLily.solver!(hp::HYPREPoisson;kwargs...)
    fill!(hp)
    HYPRE.solve!(hp.solver, hp.x, hp.A, hp.b)
    putback!(hp)
end

N = 8+2
p = zeros(N,N) #Array{Float64}(reshape(1:N^2,(N,N)))
L = ones(N,N,2); WaterLily.BC!(L,zeros(2))
σ = zeros(N,N)
pois = MultiLevelPoisson(p,L,σ)
hypre = HYPREPoisson(p,L,σ)

# matrix mult does the same, this means the coefficient are at the correct spot
x = Array{Float64}(reshape(1:N^2,(N,N)))
result = WaterLily.mult!(pois,x)
println(pois.z)
# fill the hypre vectors
hypre.r.=x; fill!(hypre)
hypre.b .= hypre.A*hypre.x
putback!(hypre) # make a field again
println(Matrix(hypre.z))
@show norm(result-hypre.z)

# solve back
# WaterLily.solver!(pois)
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



# # the classic...
# function TGV(; pow=6, Re=1000, T=Float64, mem=Array)
#     # Taylor-Green-Vortex initial velocity field
#     function u_TGV(i,x,t,ν,κ)
#         i==1 && return sin(κ*x[1])*cos(κ*x[2])*exp(-2κ^2*ν*t) # u_x
#         return  -cos(κ*x[1])*sin(κ*x[2])*exp(-2κ^2*ν*t)       # u_y
#     end
#     # Define vortex size, velocity, viscosity
#     L = 2^pow; U = 1; ν = U*L/Re
#     # make the function
#     uλ(i,xy) = u_TGV(i,xy,0,ν,2π/L)
#     # Initialize simulation
#     return Simulation((L,L), (0,0), L; U, uλ, ν, perdir=(1,2), T, mem, psolver=HYPREPoisson)
# end

# # Initialize and run
# sim = TGV()
# mom_step!(sim.flow,sim.pois)
# sim_gif!(sim,duration=10,clims=(-5,5),plotbody=true)


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
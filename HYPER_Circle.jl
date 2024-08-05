using HYPRE
using SparseArrays
using LinearAlgebra
using WaterLily; δ
using WaterLily
using Revise
using Test
include("examples/TwoD_plots.jl")

@inline WaterLily.inside(a::AbstractArray,perdir=()) = CartesianIndices(ntuple( i-> i ∈ perdir ? (1:size(a,i)) : (2:size(a,i)-1), length(axes(a))))

@fastmath @inline function filldiag(I::CartesianIndex{d},x,L,perdir) where {d}
    s = zero(eltype(L))
    for i in 1:d
        Ip = I+δ(i,I)
        (i ∈ perdir && Ip ∉ inside(x,perdir)) && (Ip = WaterLily.CIj(i,Ip,1))
        s -= @inbounds(L[I,i]+L[Ip,i])
    end
    return s
end

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
            A[J[I],J[I]] = one(T) # set all diagonals to one
        end
        N,n = WaterLily.size_u(L)
        # for I in inside(x) # fill inside the domain only otherwise we will go outside with ±δ
        #     A[J[I],J[I]] = WaterLily.diag(I,L)
        #     for i in 1:n # fill diagonal terms
        #         A[J[I-δ(i,I)],J[I]] = L[I,i]        # x[I-δ(i,I)]*L[I,i]
        #         A[J[I+δ(i,I)],J[I]] = L[I+δ(i,I),i] # x[I+δ(i,I)]*L[I+δ(i,I),i]
        #     end
        # end
        for I in inside(x,perdir)
            A[J[I],J[I]] = filldiag(I,x,L,perdir)
            for i in 1:n # fill diagonal terms
                Im = I-δ(i,I); Ip = I+δ(i,I)
                if i ∈ perdir
                   Im ∉ inside(x,perdir) && (Im = WaterLily.CIj(i,Im,N[i]))
                   Ip ∉ inside(x,perdir) && (Ip = WaterLily.CIj(i,Ip,1))
                end
                A[J[Im],J[I]] = L[I,i]  # x[I-δ(i,I)]*L[I,i]
                A[J[Ip],J[I]] = L[Ip,i] # x[I+δ(i,I)]*L[I+δ(i,I),i]
            end
        end
        # if we have a purely Neumann problem, we must fix the pressure at one spot to satisfiability
        # the system, we choose the first block
        # length(perdir) == n && (A[1,1] = one(T); A[1,2:end] .= zero(T))
        HYPRE.Init() # Init and create a conjugate gradients solver
        solver = HYPRE.GMRES(;MaxIter,Tol,PrintLevel,Logging,Precond)
        new{T,typeof(ϵ),typeof(A),typeof(x),typeof(L),typeof(solver)}(ϵ,A,r,x,L,z,solver)
    end
end

export HYPREPoisson
"""
Fill back the solution `hp.x` and `hp.z` from the solution `hp.e` and `hp.r`.
"""
function putback!(hp::HYPREPoisson)
    J = LinearIndices(hp.x)
    for I in CartesianIndices(hp.x)
        hp.x[I] = hp.ϵ[J[I]]; hp.z[I] = hp.r[J[I]]
    end
end
function Base.fill!(hp::HYPREPoisson)
    J = LinearIndices(hp.x)
    for I in CartesianIndices(hp.x) # fill the vectors entirely
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
    @WaterLily.loop hp.ϵ[I] -= mean over I ∈ CartesianIndices(hp.ϵ)
    return nothing
end

N = 8+2
p = zeros(N,N) #Array{Float64}(reshape(1:N^2,(N,N)))
L = ones(N,N,2); WaterLily.BC!(L,zeros(2))
σ = zeros(N,N)
pois = MultiLevelPoisson(copy(p),copy(L),copy(σ))
hypre = HYPREPoisson(copy(p),copy(L),copy(σ))

# matrix mult does the same, this means the coefficient are at the correct spot
x = Array{Float64}(reshape(1:N^2,(N,N)))
result = WaterLily.mult!(pois,x)
# fill the hypre vectors
hypre.z.=x; fill!(hypre)
hypre.ϵ .= hypre.A*hypre.r
putback!(hypre) # make a field again
@test all(isapprox.(result[inside(result)],hypre.x[inside(hypre.x)],atol=1e-6))

# hydrostatic pressure test case
hyrostatic!(p) = @WaterLily.inside p.z[I] = WaterLily.∂(1,I,p.L) # zero v contribution everywhere

hyrostatic!(pois)
hyrostatic!(hypre)
# solve back
WaterLily.solver!(pois;tol=1e-9)
WaterLily.solver!(hypre)
@test all(isapprox.(pois.x[inside(pois.x)],hypre.x[inside(hypre.x)],atol=1e-6))


# the classic...
function TGV(; pow=2, Re=1000, T=Float64, mem=Array)
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
    return Simulation((L,L), (0,0), L; perdir=(), U, uλ, ν, T, mem, psolver=HYPREPoisson)
end

# Initialize and run
sim = TGV();
sim_gif!(sim,duration=10,clims=(-5,5))
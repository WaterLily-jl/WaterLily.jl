"""
    exitBC!(u,u⁰,U,Δt)

Apply a 1D convection scheme to fill the ghost cell on the exit of the domain.
"""
function exitBC!(u,u⁰,Δt)
    N,_ = size_u(u)
    exitR = slice(N.-2,N[1]-1,1,3)              # exit slice excluding ghosts
    U = sum(@view(u[slice(N.-2,3,1,3),1]))/length(exitR) # inflow mass flux
    @loop u[I,1] = u⁰[I,1]-U*Δt*(u⁰[I,1]-u⁰[I-δ(1,I),1]) over I ∈ exitR
    ∮u = sum(@view(u[exitR,1]))/length(exitR)-U   # mass flux imbalance
    @loop u[I,1] -= ∮u over I ∈ exitR         # correct flux
end
"""
    perBC!(a,perdir)
Apply periodic conditions to the ghost cells of a _scalar_ field.
"""
perBC!(a,::Tuple{}) = nothing
perBC!(a, perdir, N = size(a)) = for j ∈ perdir
    @loop a[I] = a[CIj(j,I,N[j]-3)] over I ∈ slice(N,1,j)
    @loop a[I] = a[CIj(j,I,N[j]-2)] over I ∈ slice(N,2,j)
    @loop a[I] = a[CIj(j,I,3)] over I ∈ slice(N,N[j]-1,j)
    @loop a[I] = a[CIj(j,I,4)] over I ∈ slice(N,N[j],j)
end


abstract type AbstractBC{D,T,Sf,Vf,Tf} end

"""
    pressureBC!(x, bc::AbstractBC)

Hook called after the pressure Poisson solve in `project!`, with `x = flow.p`.
The default is a no-op. Subtypes of `AbstractBC` (e.g. for parallel runs) can
override this to exchange pressure ghost cells across MPI subdomain boundaries.
"""
pressureBC!(x, ::AbstractBC) = scalar_halo!(x)
pressureBC!(x, bc::AbstractBC, _) = pressureBC!(x, bc)  # default: ignore Poisson arg
"""
    BC{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Tf<:AbstractArray{T}} <: AbstractBC{D,T,Sf,Vf,Tf}

Boundary condition struct holding extending `AbstractBC`. It hold the necessary fields to impose the boundary conditions
on the `Flow.u` velocity field, as well as the boundary condition `Tuple` of `Function`, `uBC`, that needs to be passed
when creating a new `BC` struct.
"""
struct BC{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Tf<:AbstractArray{T}} <: AbstractBC{D,T,Sf,Vf,Tf}
    # BDIM fields
    V :: Vf # body velocity vector
    μ₀:: Vf # zeroth-moment vector
    μ₁:: Tf # first-moment tensor field
    σ :: Sf # scalar field as working array
    # Other BCs fields
    uBC :: Union{NTuple{D,Number},Function} # domain boundary values/function
    exitBC :: Bool # Convection exit
    perdir :: NTuple # tuple of periodic direction
    function BC(N::NTuple{D}, uBC; perdir=(), exitBC=false, mem=Array, T=Float32) where D
        Ng = N .+ 4
        Nd = (Ng..., D)
        σ, V, μ₀, μ₁ = zeros(T, Ng) |> mem, zeros(T, Nd) |> mem, ones(T, Nd) |> mem, zeros(T, Ng..., D, D) |> mem
        BC!(μ₀,ntuple(zero, D),exitBC,perdir)
        new{D,T,typeof(σ),typeof(μ₀),typeof(μ₁)}(V,μ₀,μ₁,σ,uBC,exitBC,perdir)
    end
end

"""
    ParallelBC{D,T,Sf,Vf,Tf} <: AbstractBC{D,T,Sf,Vf,Tf}

MPI-aware boundary condition for domain-decomposed WaterLily simulations.
Shares all internal arrays with the original `BC` (no copies).

Construct after building the simulation:

    sim.bc = ParallelBC(sim.bc)          # niter=1: one solve + halo exchange
    sim.bc = ParallelBC(sim.bc; niter=3) # Schwarz: 3 solve+exchange cycles

`niter` controls the number of Schwarz pressure iterations per `project!` call.
The MPI-aware `BC!`, `pressureBC!`, `residual!`, and `L₂` overrides are activated
automatically by the `WaterLilyMPIExt` extension when `ImplicitGlobalGrid` and `MPI`
are loaded.
"""
struct ParallelBC{D,T,Sf,Vf,Tf} <: AbstractBC{D,T,Sf,Vf,Tf}
    V      :: Vf
    μ₀     :: Vf
    μ₁     :: Tf
    σ      :: Sf
    uBC    :: Union{NTuple{D,Number},Function}
    exitBC :: Bool
    perdir :: NTuple
    niter  :: Int
end

"""
    ParallelBC(bc::BC; niter=1)

Wrap an existing `BC`, sharing all arrays (no copy).
Requires `ImplicitGlobalGrid` and `MPI` to be loaded (activates `WaterLilyMPIExt`).
"""
function ParallelBC(bc::BC{D,T,Sf,Vf,Tf}; niter=1) where {D,T,Sf,Vf,Tf}
    ParallelBC{D,T,Sf,Vf,Tf}(
        bc.V, bc.μ₀, bc.μ₁, bc.σ,
        bc.uBC, bc.exitBC, bc.perdir,
        niter,
    )
end

"""
    BC!(a,bc::BC,t=0)

Apply boundary conditions encapsulated in `BC` to the ghost cells of a _vector_ field `a`.
A Dirichlet condition `a[I,i]=A[i]` is applied to the vector component _normal_ to the domain boundary.
For example `aₓ(x)=Aₓ ∀ x ∈ minmax(X)`. A zero Neumann condition is applied to the tangential components.
"""
BC!(a,bc::BC,t=0) = BC!(a,bc.uBC,bc.exitBC,bc.perdir,t)
BC!(a,uBC,saveexit=false,perdir=(),t=0) = BC!(a,(i,x,t)->uBC[i],saveexit,perdir,t)
function BC!(a,uBC::Function,saveexit=false,perdir=(),t=0)
    N,n = size_u(a)
    for i ∈ 1:n, j ∈ 1:n
        if j in perdir
            @loop a[I,i] = a[CIj(j,I,N[j]-3),i] over I ∈ slice(N,1,j)
            @loop a[I,i] = a[CIj(j,I,N[j]-2),i] over I ∈ slice(N,2,j)
            @loop a[I,i] = a[CIj(j,I,3),i] over I ∈ slice(N,N[j]-1,j)
            @loop a[I,i] = a[CIj(j,I,4),i] over I ∈ slice(N,N[j],j)
        else
            if i==j # Normal direction, Dirichlet
                for s ∈ (1,2)
                    @loop a[I,i] = uBC(i,loc(i,I),t) over I ∈ slice(N,s,j)
                end
                if !saveexit || i>1
                    @loop a[I,i] = uBC(i,loc(i,I),t) over I ∈ slice(N,N[j]-1,j)
                end
                @loop a[I,i] = uBC(i,loc(i,I),t) over I ∈ slice(N,N[j],j)
            else    # Tangential directions, Neumann
                @loop a[I,i] = uBC(i,loc(i,I),t)+a[I+2δ(j,I),i]-uBC(i,loc(i,I+2δ(j,I)),t) over I ∈ slice(N,1,j)
                @loop a[I,i] = uBC(i,loc(i,I),t)+a[I+δ(j,I),i]-uBC(i,loc(i,I+δ(j,I)),t) over I ∈ slice(N,2,j)
                @loop a[I,i] = uBC(i,loc(i,I),t)+a[I-δ(j,I),i]-uBC(i,loc(i,I-δ(j,I)),t) over I ∈ slice(N,N[j]-1,j)
                @loop a[I,i] = uBC(i,loc(i,I),t)+a[I-2δ(j,I),i]-uBC(i,loc(i,I-2δ(j,I)),t) over I ∈ slice(N,N[j],j)
            end
        end
    end
end

"""
    wallBC_L!(L, perdir=())

Zero the Poisson conductivity `L` at physical (non-periodic) boundary faces.
This decouples the boundary cell from the ghost cell, giving an implicit
Neumann pressure BC (∂p/∂n = 0) at domain walls.
"""
function wallBC_L!(L, perdir=())
    N, n = size_u(L)
    for j in 1:n
        j in perdir && continue
        @loop L[I,j] = zero(eltype(L)) over I ∈ slice(N, 3, j)
    end
end

@fastmath @inline function μddn(I::CartesianIndex{np1},μ,f) where np1
    s = zero(eltype(f))
    for j ∈ 1:np1-1
        s+= @inbounds μ[I,j]*(f[I+δ(j,I)]-f[I-δ(j,I)])
    end
    return s/2
end


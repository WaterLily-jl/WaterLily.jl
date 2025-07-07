"""
    exitBC!(u,u⁰,U,Δt)

Apply a 1D convection scheme to fill the ghost cell on the exit of the domain.
"""
function exitBC!(u,u⁰,Δt)
    N,_ = size_u(u)
    exitR = slice(N.-1,N[1],1,2)              # exit slice excluding ghosts
    U = sum(@view(u[slice(N.-1,2,1,2),1]))/length(exitR) # inflow mass flux
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
    @loop a[I] = a[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
    @loop a[I] = a[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
end

abstract type AbstractBC{D,T,Sf,Vf,Tf} end
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
        Ng = N .+ 2
        Nd = (Ng..., D)
        σ, V, μ₀, μ₁ = zeros(T, Ng) |> mem, zeros(T, Nd) |> mem, ones(T, Nd) |> mem, zeros(T, Ng..., D, D) |> mem
        BC!(μ₀,ntuple(zero, D),exitBC,perdir)
        new{D,T,typeof(σ),typeof(μ₀),typeof(μ₁)}(V,μ₀,μ₁,σ,uBC,exitBC,perdir)
    end
end

"""
    BC!(a,A)

Apply boundary conditions to the ghost cells of a _vector_ field. A Dirichlet
condition `a[I,i]=A[i]` is applied to the vector component _normal_ to the domain
boundary. For example `aₓ(x)=Aₓ ∀ x ∈ minmax(X)`. A zero Neumann condition
is applied to the tangential components.
"""
BC!(a,bc::BC,t=0) = BC!(a,bc.uBC,bc.exitBC,bc.perdir,t)
BC!(a,uBC,saveexit=false,perdir=(),t=0) = BC!(a,(i,x,t)->uBC[i],saveexit,perdir,t)
function BC!(a,uBC::Function,saveexit=false,perdir=(),t=0)
    N,n = size_u(a)
    for i ∈ 1:n, j ∈ 1:n
        if j in perdir
            @loop a[I,i] = a[CIj(j,I,N[j]-1),i] over I ∈ slice(N,1,j)
            @loop a[I,i] = a[CIj(j,I,2),i] over I ∈ slice(N,N[j],j)
        else
            if i==j # Normal direction, Dirichlet
                for s ∈ (1,2)
                    @loop a[I,i] = uBC(i,loc(i,I),t) over I ∈ slice(N,s,j)
                end
                (!saveexit || i>1) && (@loop a[I,i] = uBC(i,loc(i,I),t) over I ∈ slice(N,N[j],j)) # overwrite exit
            else    # Tangential directions, Neumann
                @loop a[I,i] = uBC(i,loc(i,I),t)+a[I+δ(j,I),i]-uBC(i,loc(i,I+δ(j,I)),t) over I ∈ slice(N,1,j)
                @loop a[I,i] = uBC(i,loc(i,I),t)+a[I-δ(j,I),i]-uBC(i,loc(i,I-δ(j,I)),t) over I ∈ slice(N,N[j],j)
            end
        end
    end
end

@fastmath @inline function μddn(I::CartesianIndex{np1},μ,f) where np1
    s = zero(eltype(f))
    for j ∈ 1:np1-1
        s+= @inbounds μ[I,j]*(f[I+δ(j,I)]-f[I-δ(j,I)])
    end
    return s/2
end

# Neumann BC Building block
lowerBoundary!(r,u,Φ,ν,i,j,N,λ,::Val{false}) = @loop r[I,i] += ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) - ν*∂(j,CI(I,i),u) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν,i,j,N,λ,::Val{false}) = @loop r[I-δ(j,I),i] += -ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) + ν*∂(j,CI(I,i),u) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowerBoundary!(r,u,Φ,ν,i,j,N,λ,::Val{true}) = @loop (
    Φ[I] = ϕuP(j,CIj(j,CI(I,i),N[j]-2),CI(I,i),u,ϕ(i,CI(I,j),u),λ) -ν*∂(j,CI(I,i),u); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν,i,j,N,λ,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)